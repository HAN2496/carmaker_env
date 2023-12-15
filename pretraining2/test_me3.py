/content
****************hyper-parameters****************
epochs = 1000
batch_size = 16
soft_labels = False
adv_weight = 0
d_thresh = 0.8
z_dim = 200
z_dis = norm
model_save_step = 5
data = /content/drive/MyDrive/SocialAlgorithms_3DGAN/dataset/100_binvox_128/
device = cuda:0
g_lr = 0.0025
d_lr = 1e-05
cube_len = 128
leak_value = 0.2
bias = False
****************hyper-parameters****************
/content/drive/MyDrive/SocialAlgorithms_3DGAN/output/100_binvox_128
/content/drive/MyDrive/SocialAlgorithms_3DGAN/dataset/100_binvox_128/
data_size = 456
  0% 0/29 [00:02<?, ?it/s]
Traceback (most recent call last):
  File "/content/drive/MyDrive/SocialAlgorithms_3DGAN/./src/main.py", line 68, in <module>
    main()
  File "/content/drive/MyDrive/SocialAlgorithms_3DGAN/./src/main.py", line 62, in main
    trainer(args)
  File "/content/drive/MyDrive/SocialAlgorithms_3DGAN/src/trainer.py", line 168, in trainer
    d_total_acu = torch.mean(torch.cat((d_real_acu, d_fake_acu), se-8))
IndexError: Dimension out of range (expected to be in range of [-4, 3], but got -5)
