#!/usr/bin/python3
# -*- coding: utf-8 -*-

from datetime import datetime
import pprint
import random
import copy
import torch
import torch.nn as nn
import sys
sys.path.append("game_manager/machine_learning/")
#import omegaconf
#from hydra import compose, initialize
import os
from tensorboardX import SummaryWriter
from collections import deque
from random import random, sample,randint
import shutil
import glob 
import numpy as np
import yaml
#import subprocess

###################################################
###################################################
# ブロック操作クラス
###################################################
###################################################
class Block_Controller(object):

    # init parameter
    board_backboard = 0
    board_data_width = 0
    board_data_height = 0
    ShapeNone_index = 0
    CurrentShape_class = 0
    NextShape_class = 0

    def __init__(self):
        # init parameter
        self.mode = None
        # train
        self.init_train_parameter_flag = False
        # predict
        self.init_predict_parameter_flag = False

    ####################################
    # Yaml パラメータ読み込み
    ####################################
    def yaml_read(self,yaml_file):
        with open(yaml_file) as f:
            cfg = yaml.safe_load(f)
        return cfg

    ####################################
    # parameterを決める
    ####################################
    def set_parameter(self,yaml_file=None,predict_weight=None):
        self.result_warehouse = "outputs/"
        self.latest_dir = self.result_warehouse+"/latest"

        ########
        ## 学習の場合
        if self.mode=="train" or self.mode=="train_sample" or self.mode=="train_sample2":
            # ouput dir として日付ディレクトリ作成
            dt = datetime.now()
            self.output_dir = self.result_warehouse+ dt.strftime("%Y-%m-%d-%H-%M-%S")
            os.makedirs(self.output_dir,exist_ok=True)
            
            # weight_dir として output_dir 下に trained model フォルダを output_dir 傘下に作る
            self.weight_dir = self.output_dir+"/trained_model/"
            self.best_weight = self.weight_dir + "best_weight.pt"
            os.makedirs(self.weight_dir,exist_ok=True)
        ########
        ## 推論の場合
        else:
            dirname = os.path.dirname(predict_weight)
            self.output_dir = dirname + "/predict/"
            os.makedirs(self.output_dir,exist_ok=True)

        if yaml_file is None:
            raise Exception('Please input train_yaml file.')
        elif not os.path.exists(yaml_file):
            raise Exception('The yaml file {} is not existed.'.format(yaml_file))
        cfg = self.yaml_read(yaml_file)

        ####################
        # default.yaml を output_dir にコピーしておく
        #subprocess.run("cp config/default.yaml %s/"%(self.output_dir), shell=True)
        shutil.copy2(yaml_file, self.output_dir)

        # Tensorboard 出力フォルダ設定
        self.writer = SummaryWriter(self.output_dir+"/"+cfg["common"]["log_path"])

        ####################
        # ログファイル設定
        ########
        # 推論の場合
        if self.mode=="predict" or self.mode=="predict_sample":
            self.log = self.output_dir+"/log_predict.txt"
            self.log_score = self.output_dir+"/score_predict.txt"
            self.log_reward = self.output_dir+"/reward_predict.txt"
        ########
        # 学習の場合
        else:
            self.log = self.output_dir+"/log_train.txt"
            self.log_score = self.output_dir+"/score_train.txt"
            self.log_reward = self.output_dir+"/reward_train.txt"

        #ログ
        with open(self.log,"w") as f:
            print("start...", file=f)

        #スコアログ
        with open(self.log_score,"w") as f:
            print(0, file=f)

        #報酬ログ
        with open(self.log_reward,"w") as f:
            print(0, file=f)


        ####################
        #=====Set tetris parameter=====
        # Tetris ゲーム指定
        self.height = cfg["tetris"]["board_height"]
        self.width = cfg["tetris"]["board_width"]
        # 最大テトリミノ
        self.max_tetrominoes = cfg["tetris"]["max_tetrominoes"]
        
        ####################
        # ニューラルネットワークの入力数
        self.state_dim = cfg["state"]["dim"]
        # 学習+推論方式
        print("model name: %s"%(cfg["model"]["name"]))

        ### config/default.yaml で選択
        ## MLP の場合
        if cfg["model"]["name"]=="MLP":
            #=====load MLP=====
            # model/deepnet.py の MLP 読み込み
            from machine_learning.model.deepqnet import MLP
            # 入力数設定して MLP モデルインスタンス作成
            self.model = MLP(self.state_dim)
            # 初期状態規定
            self.initial_state = torch.FloatTensor([0 for i in range(self.state_dim)])
            #各関数規定
            self.get_next_func = self.get_next_states
            self.reward_func = self.step
            self.reward_weight = cfg["train"]["reward_weight"]
        # DQN の場合
        elif cfg["model"]["name"]=="DQN":
            #=====load Deep Q Network=====
            from machine_learning.model.deepqnet import DeepQNetwork
            # DQN モデルインスタンス作成
            self.model = DeepQNetwork()
            # 初期状態規定
            self.initial_state = torch.FloatTensor([[[0 for i in range(10)] for j in range(22)]])
            #各関数規定
            self.get_next_func = self.get_next_states_v2
            self.reward_func = self.step_v2
            self.reward_weight = cfg["train"]["reward_weight"]

        ####################
        # 推論の場合 推論ウェイトを torch　で読み込み model に入れる。
        if self.mode=="predict" or self.mode=="predict_sample":
            if not predict_weight=="None":
                if os.path.exists(predict_weight):
                    print("Load {}...".format(predict_weight))
                    # 推論インスタンス作成
                    self.model = torch.load(predict_weight)
                    # インスタンスを推論モードに切り替え
                    self.model.eval()    
                else:
                    print("{} is not existed!!".format(predict_weight))
                    exit()
            else:
                print("Please set predict_weight!!")
                exit()

        ####################
        #### finetune の場合
        #(以前の学習結果を使う場合　
        elif cfg["model"]["finetune"]:
            # weight ファイル(以前の学習ファイル)を指定
            self.ft_weight = cfg["common"]["ft_weight"]
            if not self.ft_weight is None:
                ## 読み込んでインスタンス作成
                self.model = torch.load(self.ft_weight)
                ## ログへ出力
                with open(self.log,"a") as f:
                    print("Finetuning mode\nLoad {}...".format(self.ft_weight), file=f)
                
        ## GPU 使用できるときは使う
        if torch.cuda.is_available():
            self.model.cuda()
        
        #=====Set hyper parameter=====
        #  学習バッチサイズ(学習の分割単位, データサイズを分割している)
        self.batch_size = cfg["train"]["batch_size"]
        # lr = learning rate　学習率
        self.lr = cfg["train"]["lr"]
        # pytorch 互換性のためfloat に変換
        if not isinstance(self.lr,float):
            self.lr = float(self.lr)
        # リプレイメモリサイズ
        self.replay_memory_size = cfg["train"]["replay_memory_size"]
        self.replay_memory = deque(maxlen=self.replay_memory_size)
        # 最大 Episode サイズ = 最大テトリミノ数
        # 1 Episode = 1 テトリミノ
        self.max_episode_size = self.max_tetrominoes
        self.episode_memory = deque(maxlen=self.max_episode_size)
        # 学習率減衰効果を出す EPOCH 数　(1 EPOCH = 1ゲーム)
        self.num_decay_epochs = cfg["train"]["num_decay_epochs"]
        # EPOCH 数
        self.num_epochs = cfg["train"]["num_epoch"]
        # epsilon: 過去の学習結果から変更する割合 initial は初期値、final は最終値
        # Fine Tuning 時は initial を小さめに
        self.initial_epsilon = cfg["train"]["initial_epsilon"]
        self.final_epsilon = cfg["train"]["final_epsilon"]
        # pytorch 互換性のためfloat に変換
        if not isinstance(self.final_epsilon,float):
            self.final_epsilon = float(self.final_epsilon)

        ## 損失関数（予測値と、実際の正解値の誤差）と勾配法(ADAM or SGD) の決定 
        #=====Set loss function and optimizer=====
        # ADAM の場合 .... 移動平均で振動を抑制するモーメンタム と 学習率を調整して振動を抑制するRMSProp を組み合わせている
        if cfg["train"]["optimizer"]=="Adam" or cfg["train"]["optimizer"]=="ADAM":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            self.scheduler = None
        # ADAM でない場合SGD (確率的勾配降下法、モーメンタムも STEP SIZE も学習率γもスケジューラも設定)
        else:
            # モーメンタム設定　今までの移動とこれから動くべき移動の平均をとり振動を防ぐための関数
            self.momentum =cfg["train"]["lr_momentum"] 
            # SGD に設定
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
            # 学習率更新タイミングの EPOCH 数
            self.lr_step_size = cfg["train"]["lr_step_size"]
            # 学習率γ設定　...  Step Size 進んだ EPOCH で gammma が学習率に乗算される
            self.lr_gamma = cfg["train"]["lr_gamma"]
            # スケジューラ
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_step_size , gamma=self.lr_gamma)
        # 損失算出
        self.criterion = nn.MSELoss()

        ####各パラメータ初期化
        ####=====Initialize parameter=====
        #1EPOCH ... 1試行
        self.epoch = 0
        self.score = 0
        self.max_score = -99999
        self.epoch_reward = 0
        self.cleared_lines = 0
        self.cleared_col = [0,0,0,0,0]
        self.iter = 0 
        self.state = self.initial_state 
        self.tetrominoes = 0

        # γ 割引率 = 将来の価値をどの程度下げるか
        self.gamma = cfg["train"]["gamma"]
        # 報酬を1 で正規化するかどうか、ただし消去報酬のみ　
        self.reward_clipping = cfg["train"]["reward_clipping"]

        self.score_list = cfg["tetris"]["score_list"]
        # 報酬読み込み
        self.reward_list = cfg["train"]["reward_list"]
        # Game Over 報酬 = Penalty
        self.penalty =  self.reward_list[5]

        ########
        # 報酬を 1 で正規化、ただし消去報酬のみ...Q値の急激な変動抑制
        #=====Reward clipping=====
        if self.reward_clipping:
            # 報酬リストとペナルティ(GAMEOVER 報酬)リストの絶対値の最大をとる
            self.norm_num =max(max(self.reward_list),abs(self.penalty))            
            # 最大値で割った値を改めて報酬リストとする
            self.reward_list =[r/self.norm_num for r in self.reward_list]
            # ペナルティリストも同じようにする
            self.penalty /= self.norm_num
            # max_penalty 設定と penalty 設定の小さい方を新たに ペナルティ値とする
            self.penalty = min(cfg["train"]["max_penalty"],self.penalty)

        #########
        #=====Double DQN=====
        self.double_dqn = cfg["train"]["double_dqn"]
        self.target_net = cfg["train"]["target_net"]
        if self.double_dqn:
            self.target_net = True
            

        #Target_net ON ならば
        if self.target_net:
            print("set target network...")
            self.target_model = copy.deepcopy(self.model)
            self.target_copy_intarval = cfg["train"]["target_copy_intarval"]

        ########
        #=====Prioritized Experience Replay=====
        # 優先順位つき経験学習有効ならば
        self.prioritized_replay = cfg["train"]["prioritized_replay"]
        if self.prioritized_replay:
            from machine_learning.qlearning import PRIORITIZED_EXPERIENCE_REPLAY as PER
            # 優先順位つき経験学習設定
            self.PER = PER(self.replay_memory_size, gamma=self.gamma, alpha=0.7, beta=0.5)

        ########
        #=====Multi step learning=====
        self.multi_step_learning = cfg["train"]["multi_step_learning"]
        if self.multi_step_learning:
            from machine_learning.qlearning import Multi_Step_Learning as MSL
            self.multi_step_num = cfg["train"]["multi_step_num"]
            self.MSL = MSL(step_num=self.multi_step_num,gamma=self.gamma)

    ####################################
    # リセット時にスコア計算し episode memory に penalty 追加
    ####################################
    def stack_replay_memory(self):
        if self.mode=="train" or self.mode=="train_sample" or self.mode=="train_sample2":
            self.score += self.score_list[5]
            self.episode_memory[-1][1] += self.penalty
            self.episode_memory[-1][3] = True  #store False to done lists.
            self.epoch_reward += self.penalty
            #
            if self.multi_step_learning:
                self.episode_memory = self.MSL.arrange(self.episode_memory)
                
            self.replay_memory.extend(self.episode_memory)
            self.episode_memory = deque(maxlen=self.max_episode_size)
        else:
            pass
    
    ####################################
    # Game の Reset の実施
    ####################################
    def update(self):

        if self.mode=="train" or self.mode=="train_sample" or self.mode=="train_sample2":
            # リセット時にスコア計算し episode memory に penalty 追加
            self.stack_replay_memory()

            # リプレイメモリがいっぱいでないなら、
            if len(self.replay_memory) < self.replay_memory_size / 10:
                print("================pass================")
                print("iter: {} ,meory: {}/{} , score: {}, clear line: {}, block: {}, col1-4: {}/{}/{}/{} ".format(self.iter,
                len(self.replay_memory),self.replay_memory_size / 10,self.score,self.cleared_lines
                ,self.tetrominoes, self.cleared_col[1], self.cleared_col[2], self.cleared_col[3], self.cleared_col[4]))
            # リプレイメモリがいっぱいなら
            else:
                print("================update================")
                self.epoch += 1
                # 優先順位つき経験学習有効なら
                if self.prioritized_replay:
                    # replay batch index 指定
                    batch, replay_batch_index = self.PER.sampling(self.replay_memory,self.batch_size)
                # そうでないなら
                else:
                    batch = sample(self.replay_memory, min(len(self.replay_memory),self.batch_size))
                    

                # batch から各情報を引き出す
                state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
                state_batch = torch.stack(tuple(state for state in state_batch))
                reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
                next_state_batch = torch.stack(tuple(state for state in next_state_batch))

                done_batch = torch.from_numpy(np.array(done_batch)[:, None])

                # Q 値の取得
                #max_next_state_batch = torch.stack(tuple(state for state in max_next_state_batch))
                q_values = self.model(state_batch)
                

                ###################
                # Traget net 使う場合
                if self.target_net:
                    if self.epoch %self.target_copy_intarval==0 and self.epoch>0:
                        print("target_net update...")
                        self.target_model = torch.load(self.best_weight)
                        #self.target_model = copy.copy(self.model)
                    # インスタンスを推論モードに切り替え
                    self.target_model.eval()
                    #======predict Q(S_t+1 max_a Q(s_(t+1),a))======
                    # テンソルの勾配の計算を不可
                    with torch.no_grad():
                        next_prediction_batch = self.target_model(next_state_batch)
                else:
                    # インスタンスを推論モードに切り替え
                    self.model.eval()
                    # テンソルの勾配の計算を不可とする
                    with torch.no_grad():
                        # 推論 Q値算出
                        next_prediction_batch = self.model(next_state_batch)

                ##########################
                # モデルの学習実施
                ##########################
                self.model.train()
                
                ##########################
                # Multi Step lerning の場合
                if self.multi_step_learning:
                    print("multi step learning update")
                    y_batch = self.MSL.get_y_batch(done_batch,reward_batch, next_prediction_batch)              

                # Multi Step lerning でない場合
                else:
                    y_batch = torch.cat(
                        tuple(reward if done[0] else reward + self.gamma * prediction for done ,reward, prediction in
                            zip(done_batch,reward_batch, next_prediction_batch)))[:, None]
                # 最適化対象のすべてのテンソルの勾配を 0 にする (逆伝搬backward 前に必須)
                self.optimizer.zero_grad()
                # 優先順位つき経験学習の場合
                if self.prioritized_replay:
                    # 
                    loss_weights = self.PER.update_priority(replay_batch_index,reward_batch,q_values,next_prediction_batch)
                    #print(loss_weights *nn.functional.mse_loss(q_values, y_batch))
                    loss = (loss_weights *self.criterion(q_values, y_batch)).mean()
                    #loss = self.criterion(q_values, y_batch)
                    
                    # 逆伝搬-勾配計算
                    loss.backward()
                else:
                    loss = self.criterion(q_values, y_batch)
                    # 逆伝搬-勾配計算
                    loss.backward()
                # weight を学習率に基づき更新
                self.optimizer.step()
                
                if self.scheduler!=None:
                    self.scheduler.step()
                
                log = "Epoch: {} / {}, Score: {},  block: {},  Reward: {:.4f} Cleared lines: {}, col: {}/{}/{}/{} ".format(
                    self.epoch,
                    self.num_epochs,
                    self.score,
                    self.tetrominoes,
                    self.epoch_reward,
                    self.cleared_lines,
                    self.cleared_col[1],
                    self.cleared_col[2],
                    self.cleared_col[3],
                    self.cleared_col[4]
                    )
                print(log)
                with open(self.log,"a") as f:
                    print(log, file=f)
                with open(self.log_score,"a") as f:
                    print(self.score, file=f)

                with open(self.log_reward,"a") as f:
                    print(self.epoch_reward, file=f)

                # TensorBoard への出力
                self.writer.add_scalar('Train/Score', self.score, self.epoch - 1) 
                self.writer.add_scalar('Train/Reward', self.epoch_reward, self.epoch - 1)   
                self.writer.add_scalar('Train/block', self.tetrominoes, self.epoch - 1)  
                self.writer.add_scalar('Train/clear lines', self.cleared_lines, self.epoch - 1) 

                self.writer.add_scalar('Train/1 line', self.cleared_col[1], self.epoch - 1) 
                self.writer.add_scalar('Train/2 line', self.cleared_col[2], self.epoch - 1) 
                self.writer.add_scalar('Train/3 line', self.cleared_col[3], self.epoch - 1) 
                self.writer.add_scalar('Train/4 line', self.cleared_col[4], self.epoch - 1) 

            if self.epoch > self.num_epochs:
                with open(self.log,"a") as f:
                    print("finish..", file=f)
                if os.path.exists(self.latest_dir):
                    shutil.rmtree(self.latest_dir)
                os.makedirs(self.latest_dir,exist_ok=True)
                shutil.copyfile(self.best_weight,self.latest_dir+"/best_weight.pt")
                for file in glob.glob(self.output_dir+"/*.txt"):
                    shutil.copyfile(file,self.latest_dir+"/"+os.path.basename(file))
                for file in glob.glob(self.output_dir+"/*.yaml"):
                    shutil.copyfile(file,self.latest_dir+"/"+os.path.basename(file))
                with open(self.latest_dir+"/copy_base.txt","w") as f:
                    print(self.best_weight, file=f)
                exit() 
        else:
            self.epoch += 1
            log = "Epoch: {} / {}, Score: {},  block: {}, Reward: {:.4f} Cleared lines: {}- {}/ {}/ {}/ {}".format(
            self.epoch,
            self.num_epochs,
            self.score,
            self.tetrominoes,
            self.epoch_reward,
            self.cleared_lines,
            self.cleared_col[1],
            self.cleared_col[2],
            self.cleared_col[3],
            self.cleared_col[4]
            )
        self.reset_state()
        

    ####################################
    #累積値の初期化
    ####################################
    def reset_state(self):
        if self.mode=="train" or self.mode=="train_sample" or self.mode=="train_sample2": 
            if self.score > self.max_score:
                torch.save(self.model, "{}/tetris_epoch{}_score{}.pt".format(self.weight_dir,self.epoch,self.score))
                self.max_score  =  self.score
                torch.save(self.model,self.best_weight)
        self.state = self.initial_state
        self.score = 0
        self.cleared_lines = 0
        self.cleared_col = [0,0,0,0,0]
        self.epoch_reward = 0
        self.tetrominoes = 0

    ####################################
    #削除される列を数える
    ####################################
    def check_cleared_rows(self,board):
        board_new = np.copy(board)
        lines = 0
        empty_line = np.array([0 for i in range(self.width)])
        for y in range(self.height - 1, -1, -1):
            blockCount  = np.sum(board[y])
            if blockCount == self.width:
                lines += 1
                board_new = np.delete(board_new,y,0)
                board_new = np.vstack([empty_line,board_new ])
        return lines,board_new

    ####################################
    #各列毎の高さの差(=でこぼこ度)を計算
    ####################################
    def get_bumpiness_and_height(self,board):
        # ボード上で 0 でないもの(テトリミノのあるところ)を抽出
        # (0,1,2,3,4,5,6,7) を ブロックあり True, なし Flase に変更
        mask = board != 0
        #pprint.pprint(mask, width = 61, compact = True)

        # 列方向 何かブロックががあれば、そのindexを返す
        # なければ画面ボード縦サイズを返す
        # 上記を 画面ボードの列に対して実施したの配列(長さ width)を返す
        invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), self.height)
        # 上からの距離なので反転 (配列)
        heights = self.height - invert_heights
        # 高さの合計をとる (返り値用)
        total_height = np.sum(heights)

        # 右端列を削った 高さ配列
        #currs = heights[:-1]
        currs = heights[1:-1]

        # 左端列2つを削った高さ配列
        #nexts = heights[1:]
        nexts = heights[2:]

        # 差分の絶対値をとり配列にする
        diffs = np.abs(currs - nexts)
        # 左端列は7段差まで許容
        if heights[1] - heights[0] > 3 or heights[1] - heights[0] < 0 :
            diffs = np.append(abs(heights[1] - heights[0]), diffs)

        # 差分の絶対値を合計してでこぼこ度とする
        total_bumpiness = np.sum(diffs)
        return total_bumpiness, total_height

    ####################################
    #各列の穴の個数を数える
    ####################################
    def get_holes(self, board):
        num_holes = 0
        for i in range(self.width):
            col = board[:,i]
            row = 0
            while row < self.height and col[row] == 0:
                row += 1
            num_holes += len([x for x in col[row + 1:] if x == 0])
        return num_holes

    ####################################
    # 現状状態の各種パラメータ取得 (MLP
    ####################################
    def get_state_properties(self, board):
        #削除された行の報酬
        lines_cleared, board = self.check_cleared_rows(board)
        # 穴の数
        holes = self.get_holes(board)
        # でこぼこの数
        bumpiness, height = self.get_bumpiness_and_height(board)

        return torch.FloatTensor([lines_cleared, holes, bumpiness, height])

    ####################################
    # 現状状態の各種パラメータ取得　高さ付き 今は使っていない
    ####################################
    def get_state_properties_v2(self, board):
        #削除された行の報酬
        lines_cleared, board = self.check_cleared_rows(board)
        #穴の数
        holes = self.get_holes(board)
        #でこぼこの数
        bumpiness, height = self.get_bumpiness_and_height(board)
        # 最大高さ
        max_row = self.get_max_height(board)
        return torch.FloatTensor([lines_cleared, holes, bumpiness, height,max_row])

    ####################################
    # 最大の高さを取得
    ####################################
    def get_max_height(self, board):
        sum_ = np.sum(board,axis=1)
        #print(sum_)
        row = 0
        while row < self.height and sum_[row] ==0:
            row += 1
        return self.height - row

    ####################################
    #次の状態を取得(2次元用) DQN .... 画面ボードで テトリミノ回転状態 に落下させたときの次の状態一覧を作成
    # curr_backboard 現画面
    # piece_id テトリミノ I L J T O S Z
    # currentshape_class = status["field_info"]["backboard"]
    ####################################
    def get_next_states_v2(self,curr_backboard,piece_id,CurrentShape_class):
        states = {}

        # テトリミノごとに回転数をふりわけ
        if piece_id == 5:  # O piece => 1
            num_rotations = 1
        elif piece_id == 1 or piece_id == 6 or piece_id == 7: # I, S, Z piece => 2
            num_rotations = 2
        else: # the others => 4
            num_rotations = 4

        # 回転分繰り返す
        for direction0 in range(num_rotations):
            x0Min, x0Max = self.getSearchXRange(CurrentShape_class, direction0)
            for x0 in range(x0Min, x0Max):
                # get board data, as if dropdown block
                # 画面ボードデータをコピーして指定X座標にテトリミノを固定しその画面ボードを返す
                board = self.getBoard(curr_backboard, CurrentShape_class, direction0, x0)

                # ボードを２次元化
                reshape_backboard = self.get_reshape_backboard(board)
                # numpy to tensor (配列を1次元追加)
                reshape_backboard = torch.from_numpy(reshape_backboard[np.newaxis,:,:]).float()
                # 画面ボードx0で テトリミノ回転状態 direction0 に落下させたときの次の状態を作成
                states[(x0, direction0)] = reshape_backboard
        return states

    ####################################
    #次の状態を取得(1次元用) MLP  .... 画面ボードで テトリミノ回転状態 に落下させたときの次の状態一覧を作成
    ####################################
    def get_next_states(self,curr_backboard,piece_id,CurrentShape_class):
        states = {}
        if piece_id == 5:  # O piece
            num_rotations = 1
        elif piece_id == 1 or piece_id == 6 or piece_id == 7:
            num_rotations = 2
        else:
            num_rotations = 4

        for direction0 in range(num_rotations):
            x0Min, x0Max = self.getSearchXRange(CurrentShape_class, direction0)
            for x0 in range(x0Min, x0Max):
                # get board data, as if dropdown block
                # 画面ボードデータをコピーして指定X座標にテトリミノを固定しその画面ボードを返す
                board = self.getBoard(curr_backboard, CurrentShape_class, direction0, x0)
                #ボードを２次元化
                board = self.get_reshape_backboard(board)
                states[(x0, direction0)] = self.get_state_properties(board)
        return states

    ####################################
    #ボードを２次元化
    ####################################
    def get_reshape_backboard(self,board):
        board = np.array(board)
        # 高さ, 幅で reshape
        reshape_board = board.reshape(self.height,self.width)
        # 1, 0 に変更
        reshape_board = np.where(reshape_board>0,1,0)
        return reshape_board

    ####################################
    #報酬を計算(2次元用) 
    ####################################
    def step_v2(self, curr_backboard,action,curr_shape_class):
        x0, direction0 = action
        # 画面ボードデータをコピーして指定X座標にテトリミノを固定しその画面ボードを返す
        board = self.getBoard(curr_backboard, curr_shape_class, direction0, x0)
        #ボードを２次元化
        board = self.get_reshape_backboard(board)
        ## 報酬計算元の値取得
        bampiness,height = self.get_bumpiness_and_height(board)
        max_height = self.get_max_height(board)
        hole_num = self.get_holes(board)
        lines_cleared, board = self.check_cleared_rows(board)
        ## 報酬の計算
        reward = self.reward_list[lines_cleared] 
        # 継続報酬
        reward += 0.01
        # 形状の罰
        reward -= self.reward_weight[0] *bampiness 
        if max_height > 8:
            reward -= self.reward_weight[1] * max(0,max_height)
        reward -= self.reward_weight[2] * hole_num

        self.epoch_reward += reward 

        # スコア計算
        self.score += self.score_list[lines_cleared]
        self.cleared_lines += lines_cleared
        self.cleared_col[lines_cleared] += 1
        self.tetrominoes += 1
        return reward

    ####################################
    #報酬を計算(1次元用) 
    ####################################
    def step(self, curr_backboard,action,curr_shape_class):
        x0, direction0 = action
        # 画面ボードデータをコピーして指定X座標にテトリミノを固定しその画面ボードを返す
        board = self.getBoard(curr_backboard, curr_shape_class, direction0, x0)
        #ボードを２次元化
        board = self.get_reshape_backboard(board)
        # 報酬計算元の値取得
        bampiness,height = self.get_bumpiness_and_height(board)
        max_height = self.get_max_height(board)
        hole_num = self.get_holes(board)
        lines_cleared, board = self.check_cleared_rows(board)
        #### 報酬の計算
        reward = self.reward_list[lines_cleared] 
        # 継続報酬
        reward += 0.01
        # 罰
        reward -= self.reward_weight[0] *bampiness 
        if max_height > 8:
            reward -= self.reward_weight[1] * max(0,max_height)
        reward -= self.reward_weight[2] * hole_num
        self.epoch_reward += reward

        # スコア計算
        self.score += self.score_list[lines_cleared]

        # 消した数追加
        self.cleared_lines += lines_cleared
        self.cleared_col[lines_cleared] += 1
        self.tetrominoes += 1
        return reward
           
    ####################################
    ####################################
    ####################################
    # 次の動作取得: ゲームコントローラから毎回呼ばれる
    ####################################
    ####################################
    ####################################
    ####################################
    def GetNextMove(self, nextMove, GameStatus,yaml_file=None,weight=None):

        t1 = datetime.now()
        # RESET 関数設定 callback function 代入
        nextMove["option"]["reset_callback_function_addr"] = self.update
        # mode の取得 (train である) 
        self.mode = GameStatus["judge_info"]["mode"]

        ################
        ## 初期パラメータない場合は初期パラメータ読み込み
        if self.init_train_parameter_flag == False:
            self.init_train_parameter_flag = True
            self.set_parameter(yaml_file=yaml_file,predict_weight=weight)        
        self.ind =GameStatus["block_info"]["currentShape"]["index"]
        curr_backboard = GameStatus["field_info"]["backboard"]

        ##################
        # default board definition
        self.board_data_width = GameStatus["field_info"]["width"]
        self.board_data_height = GameStatus["field_info"]["height"]

        curr_shape_class = GameStatus["block_info"]["currentShape"]["class"]
        next_shape_class= GameStatus["block_info"]["nextShape"]["class"]

        ##################
        # next shape info
        self.ShapeNone_index = GameStatus["debug_info"]["shape_info"]["shapeNone"]["index"]
        curr_piece_id =GameStatus["block_info"]["currentShape"]["index"]
        next_piece_id =GameStatus["block_info"]["nextShape"]["index"]

        ##################
        # 二次元化
        reshape_backboard = self.get_reshape_backboard(curr_backboard)
               
        #self.state = reshape_backboard


        ###################
        #画面ボードで テトリミノ回転状態 に落下させたときの次の状態一覧を作成
        # [x, directopn] xとdirectionの配列に
        next_steps =self.get_next_func(curr_backboard,curr_piece_id,curr_shape_class)
        
        ####################
        # 学習の場合
        if self.mode=="train" or self.mode=="train_sample" or self.mode=="train_sample2":
            #### init parameter
            # epsilon = 学習結果から乱数で変更する割合対象
            # num_decay_epochs より前までは比例で初期 epsilon から減らしていく
            # num_decay_ecpchs 以降は final_epsilon固定
            epsilon = self.final_epsilon + (max(self.num_decay_epochs - self.epoch, 0) * (
                    self.initial_epsilon - self.final_epsilon) / self.num_decay_epochs)
            u = random()
            # epsilon より乱数 u が小さい場合フラグをたてる
            random_action = u <= epsilon

            # 次の状態一覧の action と states で配列化
            next_actions, next_states = zip(*next_steps.items())
            # next_states のテンソルを連結
            next_states = torch.stack(next_states)

            ## GPU 使用できるときは使う
            if torch.cuda.is_available():
                next_states = next_states.cuda()
        
            ##########################
            # モデルの学習実施
            ##########################
            self.model.train()
            # テンソルの勾配の計算を不可とする(Tensor.backward() を呼び出さないことが確実な場合)
            with torch.no_grad():
                #推論Q値を算出
                predictions = self.model(next_states)[:, 0]

            # 乱数が epsilon より小さい場合は
            if random_action:
                # index を乱数とする
                index = randint(0, len(next_steps) - 1)
            else:
                # index を推論の最大値とする
                index = torch.argmax(predictions).item()
            # 次の action states を上記の index 元に決定
            next_state = next_states[index, :]
            action = next_actions[index]
            reward = self.reward_func(curr_backboard,action,curr_shape_class)
            
            done = False #game over flag
            
            #####################################
            # Double DQN 有効時
            #======predict max_a Q(s_(t+1),a)======
            #if use double dqn, predicted by main model
            if self.double_dqn:
                # 画面ボードデータをコピーして 指定X座標にテトリミノを固定しその画面ボードを返す
                next_backboard  = self.getBoard(curr_backboard, curr_shape_class, action[1], action[0])
                #画面ボードで テトリミノ回転状態 に落下させたときの次の状態一覧を作成
                next２_steps =self.get_next_func(next_backboard,next_piece_id,next_shape_class)
                # 次の状態一覧の action と states で配列化
                next2_actions, next2_states = zip(*next２_steps.items())
                # next_states のテンソルを連結
                next2_states = torch.stack(next2_states)
                ## GPU 使用できるときは使う
                if torch.cuda.is_available():
                    next2_states = next2_states.cuda()
                ##########################
                # モデルの学習実施
                ##########################
                self.model.train()
                # テンソルの勾配の計算を不可とする
                with torch.no_grad():
                    #推論Q値を算出
                    next_predictions = self.model(next2_states)[:, 0]
                # 次の index を推論の最大値とする
                next_index = torch.argmax(next_predictions).item()
                # 次の状態を index で指定し取得
                next2_state = next2_states[next_index, :]

            ################################
            # Target Next 有効時
            #if use target net, predicted by target model
            elif self.target_net:
                # 画面ボードデータをコピーして 指定X座標にテトリミノを固定しその画面ボードを返す
                next_backboard  = self.getBoard(curr_backboard, curr_shape_class, action[1], action[0])
                #画面ボードで テトリミノ回転状態 に落下させたときの次の状態一覧を作成
                next２_steps =self.get_next_func(next_backboard,next_piece_id,next_shape_class)
                # 次の状態一覧の action と states で配列化
                next2_actions, next2_states = zip(*next２_steps.items())
                next2_states = torch.stack(next2_states)
                ## GPU 使用できるときは使う
                if torch.cuda.is_available():
                    next2_states = next2_states.cuda()
                self.target_model.train()
                # テンソルの勾配の計算を不可とする
                with torch.no_grad():
                    next_predictions = self.target_model(next2_states)[:, 0]
                # 次の index を推論の最大値とする
                next_index = torch.argmax(next_predictions).item()
                # 次の状態を index で指定し取得
                next2_state = next2_states[next_index, :]
                
            #if not use target net,predicted by main model
            else:
                # 画面ボードデータをコピーして 指定X座標にテトリミノを固定しその画面ボードを返す
                next_backboard  = self.getBoard(curr_backboard, curr_shape_class, action[1], action[0])
                #画面ボードで テトリミノ回転状態 に落下させたときの次の状態一覧を作成
                next２_steps =self.get_next_func(next_backboard,next_piece_id,next_shape_class)
                # 次の状態一覧の action と states で配列化
                next2_actions, next2_states = zip(*next２_steps.items())
                # 次の状態を index で指定し取得
                next2_states = torch.stack(next2_states)

                ## GPU 使用できるときは使う
                if torch.cuda.is_available():
                    next2_states = next2_states.cuda()
                ##########################
                # モデルの学習実施
                ##########################
                self.model.train()
                # テンソルの勾配の計算を不可とする
                with torch.no_grad():
                    #推論Q値を算出
                    next_predictions = self.model(next2_states)[:, 0]

                # epsilon = 学習結果から乱数で変更する割合対象
                # num_decay_epochs より前までは比例で初期 epsilon から減らしていく
                # num_decay_ecpchs 以降は final_epsilon固定
                epsilon = self.final_epsilon + (max(self.num_decay_epochs - self.epoch, 0) * (
                self.initial_epsilon - self.final_epsilon) / self.num_decay_epochs)
                u = random()
                # epsilon より乱数 u が小さい場合フラグをたてる
                random_action = u <= epsilon
                
                # 乱数が epsilon より小さい場合は
                if random_action:
                    # index を乱数指定
                    next_index = randint(0, len(next2_steps) - 1)
                else:
                   # 次の index を推論の最大値とする
                    next_index = torch.argmax(next_predictions).item()
                # 次の状態を index により指定
                next2_state = next2_states[next_index, :]
                
            
            #=======================================
            #self.replay_memory.append([next_state, reward, next2_state,done])
            self.episode_memory.append([next_state, reward, next2_state,done])
            # 優先順位つき経験学習有効ならば
            if self.prioritized_replay:
                # 
                self.PER.store()
            
            #self.replay_memory.append([self.state, reward, next_state,done])
            nextMove["strategy"]["direction"] = action[1]
            nextMove["strategy"]["x"] = action[0]
            # Drop
            nextMove["strategy"]["y_operation"] = 1
            # ブロック落とし数
            nextMove["strategy"]["y_moveblocknum"] = 1

            # 1ゲーム(EPOCH)の上限テトリミノ数を超えたらリセットフラグを立てる
            if self.tetrominoes > self.max_tetrominoes:
                nextMove["option"]["force_reset_field"] = True
            # STATE = NEXT へ
            self.state = next_state

        ####################
        # 推論
        elif self.mode == "predict" or self.mode == "predict_sample":
            #推論モードに切り替え
            self.model.eval()
            # 画面ボードの次の状態一覧を action と states にわける
            next_actions, next_states = zip(*next_steps.items())
            next_states = torch.stack(next_states)
            #推論Q値を算出
            predictions = self.model(next_states)[:, 0]
            # 最大値の index 取得
            index = torch.argmax(predictions).item()
            # 次の action を index を元に決定
            action = next_actions[index]
            nextMove["strategy"]["direction"] = action[1]
            nextMove["strategy"]["x"] = action[0]
            nextMove["strategy"]["y_operation"] = 1
            nextMove["strategy"]["y_moveblocknum"] = 1
        return nextMove
    
    ####################################
    # 
    # self,
    # Shape_class: 現在と予告テトリミノの配列
    # direction: 現在のテトリミノ方向
    ####################################
    def getSearchXRange(self, Shape_class, direction):
        #
        # get x range from shape direction.
        #
        minX, maxX, _, _ = Shape_class.getBoundingOffsets(direction) # get shape x offsets[minX,maxX] as relative value.
        xMin = -1 * minX
        xMax = self.board_data_width - maxX
        return xMin, xMax

    ####################################
    # direction (回転状態)のテトリミノ座標配列を取得し、それをx,yに配置した場合の座標配列を返す
    ####################################
    def getShapeCoordArray(self, Shape_class, direction, x, y):
        #
        # get coordinate array by given shape.
        # direction (回転状態)のテトリミノ座標配列を取得し、それをx,yに配置した場合の座標配列を返す
        coordArray = Shape_class.getCoords(direction, x, y) # get array from shape direction, x, y.
        return coordArray

    ####################################
    # 画面ボードデータをコピーして指定X座標にテトリミノを固定しその画面ボードを返す
    # board_backboard: 現状画面ボード
    # Shape_class: テトリミノ現/予告リスト
    # direction: テトリミノ回転方向
    # x: テトリミノx座標
    ####################################
    def getBoard(self, board_backboard, Shape_class, direction, x):
        # 
        # get new board.
        #
        # copy backboard data to make new board.
        # if not, original backboard data will be updated later.
        board = copy.deepcopy(board_backboard)
        # 指定X座標にテトリミノを固定しその画面ボードを返す
        _board = self.dropDown(board, Shape_class, direction, x)
        return _board

    ####################################
    # 指定X座標にテトリミノを固定しその画面ボードを返す
    # board: 現状画面ボード
    # Shape_class: テトリミノ現/予告リスト
    # direction: テトリミノ回転方向
    # x: テトリミノx座標
    ####################################
    def dropDown(self, board, Shape_class, direction, x):
        # 
        # internal function of getBoard.
        # -- drop down the shape on the board.
        # 

        # 画面ボード下限座標として dy 設定
        dy = self.board_data_height - 1
        # direction (回転状態)のテトリミノ座標配列を取得し、それをx,yに配置した場合の座標配列を返す
        coordArray = self.getShapeCoordArray(Shape_class, direction, x, 0)
        # update dy
        # テトリミノ座標配列ごとに...
        for _x, _y in coordArray:
            _yy = 0
            # _yy を一つずつ落とすことによりブロックの落下下限を確認
            # _yy+テトリミノ座標y が 画面下限より上　かつ　(_yy +テトリミノ座標yが画面外 または テトリミノ座標_x,_yy+テトリミノ座標_yのブロックがない)
            while _yy + _y < self.board_data_height and (_yy + _y < 0 or board[(_y + _yy) * self.board_data_width + _x] == self.ShapeNone_index):
                #_yy を足していく(下げていく)
                _yy += 1
            _yy -= 1
            # 下限座標/今までの下限より小さいなら __yy を落下下限として設定
            if _yy < dy:
                dy = _yy
        # dy: テトリミノ落下下限座標を指定
        _board = self.dropDownWithDy(board, Shape_class, direction, x, dy)
        return _board

    ####################################
    # 指定位置にテトリミノを固定する
    # board: 現状画面ボード
    # Shape_class: テトリミノ現/予告リスト
    # direction: テトリミノ回転方向
    # x: テトリミノx座標
    # dy: テトリミノ落下下限座標を指定
    ####################################
    def dropDownWithDy(self, board, Shape_class, direction, x, dy):
        #
        # internal function of dropDown.
        #
        _board = board
        # direction (回転状態)のテトリミノ座標配列を取得し、それをx,yに配置した場合の座標配列を返す
        coordArray = self.getShapeCoordArray(Shape_class, direction, x, 0)
        # テトリミノ座標配列を順に進める
        for _x, _y in coordArray:
            #dy の落下下限の 画面ボードにブロックを配置してき、その画面ボードデータを返す
            _board[(_y + dy) * self.board_data_width + _x] = Shape_class.shape
        return _board
BLOCK_CONTROLLER_TRAIN = Block_Controller()
