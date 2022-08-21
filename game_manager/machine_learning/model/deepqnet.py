import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,input_dim):
        #親の nn.Module の初期化を継承実行
        super(MLP, self).__init__()

        #####################
        # 順伝搬処理初期化
        # Linear 線形変換
        # 活性化関数: ReLU = 正規化線形関数, 負の数は扱えないので注意、今回はすべて正の数なのでOK
        # 入力 input_dim, 中間層 64 固定, 層数 3
        self.conv1 = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Linear(64, 1))

        # 重みづけ、バイアス初期化
        self._create_weights()

    #####################
    ## 重みづけ、バイアス初期化
    def _create_weights(self):
        ## nn をすべての module をイテレート
        for m in self.modules():
            ## nn.Linear (行列) のインスタンスか?
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    ##########################
    # 順伝搬処理
    # PyTorchはnn.Moduleクラスを基底とし、順伝搬の処理をforwardの中に書いている
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
    
class DeepQNetwork(nn.Module):
    def __init__(self):
        #親の nn.Module の初期化を継承実行
        super(DeepQNetwork, self).__init__()
        #####################
        # 順伝搬処理初期化
        # conv2d 二次元変換
        # 活性化関数: ReLU = 正規化線形関数, 負の数は扱えないので注意、今回はすべて正の数なのでOK
        # 入力 1 中間層 32 固定, 層数 3
        # kernel_size (4x4), 
        # padding1 で周囲1マスの演算もする
        # stride は 2マスずらしで演算
        self.conv1 = nn.Sequential(
                nn.Conv2d(1,32, kernel_size=4, stride=2,padding=1,
                padding_mode='zeros',bias=False),
                nn.ReLU())
        # 周囲を 0 で Padding
        # Kernel_size 5 にし、 stride を1で演算
        self.conv2 = nn.Sequential(
                nn.ConstantPad2d((2,2,2,2),0),
                nn.Conv2d(32, 32, kernel_size=5, stride=1,padding=0,
                bias=False),
                nn.ReLU())
        
        self.conv3 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=5, stride=2, 
                bias=False, padding_mode='zeros'),
                nn.ReLU())

        ####################
        ## 列規定
        self.num_feature = 64*4*1
        ## 線形変換 (活性化関数なし)
        self.fc1 = nn.Sequential(nn.Linear(self.num_feature,256))
        self.fc2 = nn.Sequential(nn.Linear(256,256))
        self.fc3 = nn.Sequential(nn.Linear(256,1))
        
        # 重みづけ、バイアス初期化
        self._create_weights()

    #####################
    ## 重みづけ、バイアス初期化
    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    ##############################
    # 順伝搬処理
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)    
        ## テンソルの列数の調整 (num_feature = 256 に合わせる)
        x = x.view(-1, self.num_feature ) 
        ## 
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x