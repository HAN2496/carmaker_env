#@title 3DGAN training (64)

gan_folder = "/content/drive/MyDrive/SocialAlgorithms_3DGAN" #@param {type:"string"}
logs = "100_binvox_128" #@param {type:"string"}
epochs = 1000 #@param {type:"integer"}
batch_size=16 #@param {type:"integer"}
model_save_steps=100 #@param {type:"integer"}
model_name = "100_binvox_128" #@param {type:"string"}
cube_size = 128 #@param {type:"integer"}
data_dir = "/content/drive/MyDrive/SocialAlgorithms_3DGAN/dataset/100_binvox_128/" #@param {type:"string"}
output_dir = "/content/drive/MyDrive/SocialAlgorithms_3DGAN/output" #@param {type:"string"}

%cd $gan_folder

!python ./src/main.py \
--logs=$logs \
--epochs=$epochs \
--batch_size=$batch_size \
--model_save_steps=$model_save_steps \
--model_name=$model_name \
--data_dir=$data_dir \
--cube_len=$cube_size \
--output_dir=$output_dir
