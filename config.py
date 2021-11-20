import os

root_path = os.path.abspath(os.path.dirname(__file__))
script_embedding_path = os.path.join(root_path, 'data\\script_scene_embedding.npy')
max_length = 430
vocab_path = os.path.join(root_path, 'lib\\nezha-base-www\\vocab.txt')
all_content = os.path.join(root_path, 'data\\all_content.txt')
model_dir = os.path.join(root_path, 'lib\\chinese-roberta-wwm-ext')
model_path = os.path.join(root_path, 'lib\\chinese-roberta-wwm-ext\\pytorch_model.bin')
nezha_pretrain_model_path = os.path.join(root_path, 'model\\nezha_base_wwm_pretrain\\nezha_base_wwm.bin')
nezha_pretrain_model_dir = os.path.join(root_path, 'model\\nezha_base_wwm_pretrain\\')
task_by_pretrain = os.path.join(root_path, 'model\\task_by_pretrain')
task_by_prompt = os.path.join(root_path, 'model\\task_by_prompt')
mlm_probability = 0.15
batch_size = 16
device = 'cuda'
lr = 3e-6
# lr = 0.01
max_lr = 0.1
gradient_accumulation = 4
max_grad_norm = 1.0
train_data_path = os.path.join(root_path, 'data\\train_dataset_v2.tsv')
test_data_path = os.path.join(root_path, 'data\\test_dataset.tsv')
emotions2id = {'在乎': 0, '开心': 1, '惊讶': 2, '愤怒': 3, '恐惧': 4, '悲伤': 5}
# id2emotions = {0: '在乎', 1: '开心', 2: '惊讶', 3: '愤怒', 4: '恐惧', 5: '悲伤'}
id2emotions = {0: '爱', 1: '乐', 2: '惊', 3: '怒', 4: '恐', 5: '哀'}
emotion_prompt = {0: '没', 1: '稍', 2: '较', 3: '很'}
emotion0_prompt_train_path = os.path.join(root_path, 'data\\emotion0_prompt_train.txt')
emotion1_prompt_train_path = os.path.join(root_path, 'data\\emotion1_prompt_train.txt')
emotion2_prompt_train_path = os.path.join(root_path, 'data\\emotion2_prompt_train.txt')
emotion3_prompt_train_path = os.path.join(root_path, 'data\\emotion3_prompt_train.txt')
emotion4_prompt_train_path = os.path.join(root_path, 'data\\emotion4_prompt_train.txt')
emotion5_prompt_train_path = os.path.join(root_path, 'data\\emotion5_prompt_train.txt')

emotion0_prompt_test_path = os.path.join(root_path, 'data\\emotion0_prompt_test.txt')
emotion1_prompt_test_path = os.path.join(root_path, 'data\\emotion1_prompt_test.txt')
emotion2_prompt_test_path = os.path.join(root_path, 'data\\emotion2_prompt_test.txt')
emotion3_prompt_test_path = os.path.join(root_path, 'data\\emotion3_prompt_test.txt')
emotion4_prompt_test_path = os.path.join(root_path, 'data\\emotion4_prompt_test.txt')
emotion5_prompt_test_path = os.path.join(root_path, 'data\\emotion5_prompt_test.txt')
seed = 354689210
emotion_label_num = 24
label_weight = [6.217190728214417, 417.6273764258555, 634.8901734104046, 539.7346437346438, 6.488804867962427,
                107.0004870920604, 460.52830188679246, 963.4736842105264, 6.383958151700087, 165.0428249436514,
                371.6954314720812, 784.5428571428571, 6.683867826933609, 110.55460493205838, 185.53378378378378,
                382.0382608695652, 6.521940502345466, 143.76439790575915, 219.8918918918919, 545.091811414392,
                7.117187753118419, 78.0639658848614, 111.39553752535497, 228.58688865764827]
sample_weight_path = os.path.join(root_path, 'data\\sample_weight.txt')

emotion0_model_path = os.path.join(root_path, 'model\\emotion0\\model.bin')
emotion1_model_path = os.path.join(root_path, 'model\\emotion1\\model.bin')
emotion2_model_path = os.path.join(root_path, 'model\\emotion2\\model.bin')
emotion3_model_path = os.path.join(root_path, 'model\\emotion3\\model.bin')
emotion4_model_path = os.path.join(root_path, 'model\\emotion4\\model.bin')
emotion5_model_path = os.path.join(root_path, 'model\\emotion5\\model.bin')

# emotion0_label_weight = [1.0362, 690.6046, 1050.8150, 890.9558]
# emotion1_label_weight = [1.0815, 170.8334, 760.7547, 160.5789]
# emotion2_label_weight = [1.0640, 127.5071, 61.9492, 130.7571]
# emotion3_label_weight = [1.1140, 180.4258, 300.9223, 163.6730]
# emotion4_label_weight = [1.0870, 230.9607, 360.6486, 900.8486]
# emotion5_label_weight = [1.1862, 13.0107, 18.5659, 38.0978]

emotion0_label_weight = [1.0362, 690.6046, 1050.8150, 890.9558]
emotion1_label_weight = [1.0815, 170.8334, 760.7547, 160.5789]
emotion2_label_weight = [1.0640, 1270.5071, 610.9492, 1300.7571]
emotion3_label_weight = [1.1140, 1800.4258, 3000.9223, 1630.6730]
emotion4_label_weight = [1.0870, 2300.9607, 3600.6486, 9000.8486]
emotion5_label_weight = [1.1862, 130.0107, 180.5659, 380.0978]

# emotion0_label_weight = [0.8392, 0.9976, 0.9984, 0.9981]
# emotion1_label_weight = [0.8459, 0.9907, 0.9978, 0.9990]
# emotion2_label_weight = [0.8434, 0.9939, 0.9973, 0.9987]
# emotion3_label_weight = [0.8504, 0.9910, 0.9946, 0.9974]
# emotion4_label_weight = [0.8467, 0.9930, 0.9955, 0.9982]
# emotion5_label_weight = [0.8595, 0.9872, 0.9910, 0.9956]

# emotion0_label_weight = [1, 1, 1, 1]
# emotion1_label_weight = [1, 1, 1, 1]
# emotion2_label_weight = [1, 1, 1, 1]
# emotion3_label_weight = [1, 1, 1, 1]
# emotion4_label_weight = [1, 1, 1, 1]
# emotion5_label_weight = [1, 1, 1, 1]


count_label = [35333, 526, 346, 407, 33854, 2053, 477, 228, 34410, 1331, 591, 280, 32866, 1987, 1184, 575, 33682, 1528,
               999, 403, 30865, 2814, 1972, 961]

submit_path = os.path.join(root_path, "output\\test\\{}_submit".format(model_dir.split('\\')[-1]))
script_scene_id_path = os.path.join(root_path, 'data\\script_scene_id.txt')
nezha_26199_model_dir = os.path.join(root_path, 'lib\\nezha-vocab_26199')

punctuation = ["。", "；", "？", "：", "！", "，"]

'''
emotion0[-0.20133739709854126, 0.38420259952545166]
emotion1[-0.3107595145702362, 0.4048469364643097]
emotion2[-0.687850832939148, 0.6575840711593628]
emotion3[-0.5158796906471252, 0.8605697751045227]
emotion4[-0.46787211298942566, 0.630348265171051]
emotion5[-0.3218633532524109, 0.7620697617530823]

1.5tanh + 1.5
emotion0[0.048329830169677734, 2.296881914138794]
emotion1[0.05573713779449463, 1.7135957479476929]
emotion2[0.03969621658325195, 1.9341607093811035]
emotion3[0.06589066982269287, 2.0279622077941895]
emotion4[0.04510807991027832, 1.6044617891311646]
emotion5[-0.3218633532524109, 0.7620697617530823]
'''

