import torch.nn as nn
import torch



class SA_Attn_Mem(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.layer_q = nn.Conv2d(input_dim, hidden_dim, (1, 1))
        self.layer_k = nn.Conv2d(input_dim, hidden_dim, (1, 1))
        self.layer_k2 = nn.Conv2d(input_dim, hidden_dim, (1, 1))

        self.layer_v = nn.Conv2d(input_dim, input_dim, (1, 1))
        self.layer_v2 = nn.Conv2d(input_dim, input_dim, (1, 1))

        self.layer_z = nn.Conv2d(input_dim * 2, input_dim * 2, (1, 1))
        self.layer_m = nn.Conv2d(input_dim * 3, input_dim * 3, (1, 1))

    def forward(self, h, m):
        batch_size, channels, H, W = h.shape
        # **********************  feature aggregation ******************** #

        # Use 1x1 convolution for Q,K,V Generation
        K_h = self.layer_k(h)
        K_h = K_h.view(batch_size, self.hidden_dim, H * W)

        Q_h = self.layer_q(h)
        Q_h = Q_h.view(batch_size, self.hidden_dim, H * W)
        Q_h = Q_h.transpose(1, 2)

        V_h = self.layer_v(h)
        V_h = V_h.view(batch_size, self.input_dim, H * W)

        K_m = self.layer_k2(m)
        K_m = K_m.view(batch_size, self.hidden_dim, H * W)

        V_m = self.layer_v2(m)
        V_m = V_m.view(batch_size, self.input_dim, H * W)

        # **********************  hidden h attention ******************** #
        # [batch_size,H*W,H*W]
        A_h = torch.softmax(torch.bmm(Q_h, K_h), dim=-1)

        Z_h = torch.matmul(A_h, V_h.permute(0, 2, 1))
        Z_h = Z_h.transpose(1, 2).view(batch_size, self.input_dim, H, W)
        # **********************  memory m attention ******************** #
        # [batch_size,H*W,H*W]
        A_m = torch.softmax(torch.bmm(Q_h, K_m), dim=-1)

        Z_m = torch.matmul(A_m, V_m.permute(0, 2, 1))
        Z_m = Z_m.transpose(1, 2).view(batch_size, self.input_dim, H, W)

        W_z = torch.cat([Z_h, Z_m], dim=1)
        Z = self.layer_z(W_z)   # [batch_size,in_channels*2,H,W]

        # Memory Updating
        combined = self.layer_m(torch.cat([Z, h], dim=1))
        mo, mg, mi = torch.split(combined, self.input_dim, dim=1)
        #
        mi = torch.sigmoid(mi)
        new_m = (1 - mi) * m + mi * torch.tanh(mg)
        new_h = torch.sigmoid(mo) * new_m

        return new_h, new_m


class SAConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, att_hidden_dim, kernel_size, bias):
        """
           Initialize SA ConvLSTM cell.
           Parameters
           ---------
           input_dim: int
               Number of channels of input tensor.
           hidden_dim: int
               Number of channels of hidden state.
           kernel_size: (int, int)
               Size of the convolutional kernel.
           bias: bool
               Whether to add the bias.
           att_hidden_dim: int
               Number of channels of attention hidden state
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.attention_layer = SA_Attn_Mem(hidden_dim, att_hidden_dim)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                      out_channels=4 * self.hidden_dim,
                      kernel_size=kernel_size,
                      padding=self.padding,
                      bias=self.bias),
            nn.GroupNorm(4 * hidden_dim, 4 * hidden_dim)
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur, m_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        h_next, m_next = self.attention_layer(h_next, m_cur)
        return h_next, c_next, m_next

    # initialize h, c, m
    def init_hidden(self, batch_size, image_size,device):
        height, width = image_size
        h = torch.zeros(batch_size, self.hidden_dim, height, width,device=device)
        c = torch.zeros(batch_size, self.hidden_dim, height, width,device=device)
        m = torch.zeros(batch_size, self.hidden_dim, height, width,device=device)
        return h, c, m


class SAConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, attn_hidden_dim, kernel_size, num_layers, batch_first=False, bias=True,
                 return_all_layers=False):
        super().__init__()
        self._check_kernel_size_consistency(kernel_size)
        # make sure that both "kernel_size" and 'hidden_dim' are lists having len=num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        attn_hidden_dim = self._extend_for_multilayer(attn_hidden_dim, num_layers)

        if not len(kernel_size) == len(hidden_dim) == len(attn_hidden_dim) == num_layers:
            raise ValueError("Inconsistent list length.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.attn_hidden_dim = attn_hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(SAConvLSTMCell(input_dim=cur_input_dim,
                                            hidden_dim=self.hidden_dim[i],
                                            kernel_size=self.kernel_size[i],
                                            att_hidden_dim=self.attn_hidden_dim[i],
                                            bias=self.bias,
                                            ))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            # (t,b,c,h,w)->(b,t,c,h,w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        b, _, _, h, w = input_tensor.size()
        if hidden_state is not None:
            raise NotImplementedError
        else:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w),device=input_tensor.device)
        layer_output_list = []
        last_state_list = []
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):
            h, c, m = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c, m = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c, m])
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h, c, m])
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size,device):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size,device))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (
                isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))
        ):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param



class SE_Block(nn.Module):
    def __init__(self, ch_in, ratio=2):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // ratio, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        z = x * y.expand_as(x)

        return z


class Alx(nn.Module):
    def __init__(self, init_weights=False):
        super(Alx, self).__init__()
        # 用nn.Sequential()将网络打包成一个模块，精简代码
        self.features = nn.Sequential(  # 卷积层提取图像特征
            nn.Conv2d(1, 30, kernel_size=5, stride=2, padding=2),  # (, 1, 224, 224)  (, 3, 111, 111)
            nn.BatchNorm2d(30, affine=True),
            nn.ReLU(inplace=True),  # 直接修改覆盖原值，节省运算内存
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(30, 60, kernel_size=3, padding=1, groups=10),  # ( , 5, 54, 54)
            nn.BatchNorm2d(60, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(60, 180, kernel_size=3, padding=1, groups=10),  # ( , 7, 28, 28)
            nn.BatchNorm2d(180, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (, 7, 27 27)
            nn.Conv2d(180, 250, kernel_size=3, padding=1, groups=10),  # output[128, 13, 13]
            nn.BatchNorm2d(250, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(250, 300, kernel_size=3, padding=1, groups=10),  # output[128, 13, 13]
        )
        if init_weights:       # 如果init_weights=True的话就调用初始话权重函数
            self._initialize_weights()
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.conv1 = nn.Conv2d(300, 5, kernel_size=1)
        self.BE = SE_Block(300)

    # 前向传播过程
    def forward(self, x):
        x0 = self.features(x)
        x0 = self.BE(x0)
        x0 = self.conv1(x0)
        x0 = self.avgpool(x0)

        return x0


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class ConvLSTMNet(nn.Module):
    def __init__(self, init_weights=False):
        super(ConvLSTMNet, self).__init__()
        self.branch1 = Alx()
        self.branch2 = SAConvLSTM(input_dim=5,
                                  hidden_dim=[15],
                                  attn_hidden_dim=15,
                                  kernel_size=(3, 3),
                                  num_layers=1,
                                  batch_first=True,
                                  bias=True,
                                  return_all_layers=False)  # ( , 9, 13, 13)
        self.branch3 = SAConvLSTM(input_dim=15,
                                  hidden_dim=[30],
                                  attn_hidden_dim=30,
                                  kernel_size=(3, 3),
                                  num_layers=1,
                                  batch_first=True,
                                  bias=True,
                                  return_all_layers=False)

        self.SE = SE_Block(5, 2)

        if init_weights:
            self._initialize_weights()


    def forward(self, x):
        x = x.reshape(-1, 1, 112, 112)
        x = self.branch1(x)
        x = x.reshape(-1, 5, 5, 6, 6)
        _, x0 = self.branch2(x)
        x0 = x0[0][0]
        x0 = x0.reshape(-1, 30, 15, 6, 6)
        _, x1 = self.branch3(x0)
        x1 = x1[0][0]


        return x1


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class AlexNet(nn.Module):
    def __init__(self, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(150, 120, kernel_size=5, stride=2, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.BatchNorm2d(120, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[48, 27, 27]
            nn.Conv2d(120, 270, kernel_size=5, padding=2),  # output[128, 27, 27]
            nn.BatchNorm2d(270, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 13, 13]
            nn.Conv2d(270, 400, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.BatchNorm2d(400, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(400, 400, kernel_size=3, padding=1),  # output[192, 13, 13]

        )
        self.SE1 = SE_Block(ch_in=400, ratio=2)
        self.conv1 = nn.Conv2d(400, 270, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.SE1(x)
        x = self.conv1(x)
        x1 = self.avgpool(x)

        return x1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class CNN_ConvLSTMNet(nn.Module):
    def __init__(self, init_weights=False):
        super(CNN_ConvLSTMNet, self).__init__()
        self.branch1 = AlexNet()
        self.branch2 = ConvLSTMNet()
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = torch.cat((x1, x2), 1)

        return x3

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)




class MFSTNet(nn.Module):
    def __init__(self, init_weights=False):
        super(MFSTNet, self).__init__()
        self.branch1 = CNN_ConvLSTMNet()
        self.branch2 = SAConv()
        self.bn1 = nn.BatchNorm1d(330*6*6, affine=True)
        self.BE = SE_Block(330)
        self.design_fc = nn.Sequential(
            nn.Linear(330*6*6, 4080),
            nn.BatchNorm1d(4080),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4080, 1080),
            nn.BatchNorm1d(1080),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1080, 3),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, input_1_train, input_2_train):
        x1 = self.branch1(input_1_train)
        x2 = self.branch2(input_2_train)
        x3 = torch.cat((x1, x2), 1)
        x3 = self.BE(x3)
        x4 = x3.view(x3.size(0), -1)
        x4 = self.bn1(x4)
        x4 = self.design_fc(x4)

        return x4



class SAConv(nn.Module):
    def __init__(self, init_weights=False):
        super(SAConv, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(4, 30, kernel_size=5, stride=2, padding=2),
            SE_Block(30),
            nn.Conv2d(30, 4, kernel_size=1),
            nn.AdaptiveAvgPool2d((6, 6))
        )

        self.branch2 = SAConvLSTM(input_dim=4,
                                  hidden_dim=[30],
                                  attn_hidden_dim=[30],
                                  kernel_size=(3, 3),
                                  num_layers=1,
                                  batch_first=True,
                                  bias=True,
                                  return_all_layers=False)   # ( , 9, 13, 13)

        self.bn1 = nn.BatchNorm1d(30*6*6, affine=True)  # 5*3*3
        self.design_fc = nn.Sequential(
            nn.Linear(30*6*6, 3),
        )
        if init_weights:
            self._initialize_weights()

    # 前向传播过程
    def forward(self, x):
        x = x.reshape(-1, 4, 112, 112)
        x = self.branch1(x)
        x = x.reshape(-1, 30, 4, 6, 6)
        _, x0 = self.branch2(x)
        x0 = x0[0][0]

        return x0

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class FDAFNet(nn.Module):
    def __init__(self, init_weights=False):
        super(FDAFNet, self).__init__()
        self.branch1 = CNN_ConvLSTMNet()
        self.branch2 = SAConv()
        self.bn1 = nn.BatchNorm1d(40*6*6, affine=True)
        self.BE = SE_Block(40)
        self.design_fc = nn.Sequential(
            nn.Linear(40*6*6, 3),
        )
        if init_weights:
            self._initialize_weights()


    def forward(self, input_1_train, input_2_train):
        x1 = self.branch1(input_1_train)
        x2 = self.branch2(input_2_train)
        x3 = x1 + x2  # [80, 12, 12]
        x3 = self.BE(x3)
        x4 = x3.view(x3.size(0), -1)
        x4 = self.bn1(x4)
        x4 = self.design_fc(x4)

        return x4
