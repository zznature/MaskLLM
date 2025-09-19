# Llama8b稀疏训练日志

traing script: `llama8b_scripts/llama8b_presparse_training_tp8.sh`

```bash
# 设置GPU可见性
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 选项1: 稀疏化训练 (基于已生成的预稀疏模型) 从预稀疏checkpoint开始
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8.sh 0

# 继续训练 从训练checkpoint恢复
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8.sh 1
```

## 1st training

stop at iteration 850 for NaN error in gradient
log-file: `output/logs/llama8b_presparse_training/training_date_25-09-02_time_04-47-14.log`

```bash
# evaluation result
validation loss at iteration 100 | lm loss value: 1.255308E+01 | lm loss PPL: 2.829645E+05 | reg loss value: 7.631986E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 200 | lm loss value: 1.535134E+01 | lm loss PPL: 4.645191E+06 | reg loss value: 7.659308E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 300 | lm loss value: 1.097285E+01 | lm loss PPL: 5.827020E+04 | reg loss value: 7.681837E+03 | reg loss PPL: 4.851652E+08 |   
```

## 2nd training

- changed some parameters to make the training more stable

| 参数类别 | 1st training | 2nd training | 修改幅度 | 修改位置 |
|---------|---------|---------|---------|---------|
| 学习率 | 5e-5 | 2e-5 | ⬇️ 2.5x | 第128行 |
| 最小学习率 | 5e-6 | 2e-6 | ⬇️ 2.5x | 第129行 |
| 正则化 | 1e-5 | 2e-6 | ⬇️ 5x | 第103行 |
| Gumbel缩放 | 1e2 5e2 | 5e1 2.5e2 | ⬇️ 2x | 第103行 |
| Warmup步数 | 0 | 400 | ➕ 新增 | 第93行 |
| 保存间隔 | 500 | 200 | ⬆️ 2.5x频率 | 第90行 |
| 评估间隔 | 100 | 50 | ⬆️ 2x频率 | 第91行 |

log-file: `output/logs/llama8b_presparse_training/training_date_25-09-05_time_02-51-50.log`

```bash
# evaluation result
validation loss at iteration 50 | lm loss value: 1.161736E+01 | lm loss PPL: 1.110081E+05 | reg loss value: 7.615785E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 100 | lm loss value: 1.340704E+01 | lm loss PPL: 6.646664E+05 | reg loss value: 7.616810E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 150 | lm loss value: 1.296712E+01 | lm loss PPL: 4.281035E+05 | reg loss value: 7.619352E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 200 | lm loss value: 1.549217E+01 | lm loss PPL: 5.347674E+06 | reg loss value: 7.622215E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 250 | lm loss value: 1.176644E+01 | lm loss PPL: 1.288542E+05 | reg loss value: 7.625785E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 300 | lm loss value: 1.128546E+01 | lm loss PPL: 7.965480E+04 | reg loss value: 7.631141E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 350 | lm loss value: 1.188648E+01 | lm loss PPL: 1.452885E+05 | reg loss value: 7.633798E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 400 | lm loss value: 1.250351E+01 | lm loss PPL: 2.692799E+05 | reg loss value: 7.635353E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 450 | lm loss value: 1.075681E+01 | lm loss PPL: 4.694890E+04 | reg loss value: 7.641146E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 500 | lm loss value: 1.104297E+01 | lm loss PPL: 6.250275E+04 | reg loss value: 7.644854E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 550 | lm loss value: 1.113715E+01 | lm loss PPL: 6.867557E+04 | reg loss value: 7.648693E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 600 | lm loss value: 1.134056E+01 | lm loss PPL: 8.416755E+04 | reg loss value: 7.653197E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 650 | lm loss value: 1.179350E+01 | lm loss PPL: 1.323889E+05 | reg loss value: 7.659308E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 700 | lm loss value: 1.107539E+01 | lm loss PPL: 6.456237E+04 | reg loss value: 7.661159E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 750 | lm loss value: 1.115774E+01 | lm loss PPL: 7.010438E+04 | reg loss value: 7.663946E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 800 | lm loss value: 1.116398E+01 | lm loss PPL: 7.054326E+04 | reg loss value: 7.671310E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 850 | lm loss value: 1.126647E+01 | lm loss PPL: 7.815661E+04 | reg loss value: 7.674691E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 900 | lm loss value: 1.048755E+01 | lm loss PPL: 3.586627E+04 | reg loss value: 7.679147E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 950 | lm loss value: 1.047793E+01 | lm loss PPL: 3.552287E+04 | reg loss value: 7.681950E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 1000 | lm loss value: 1.017551E+01 | lm loss PPL: 2.625237E+04 | reg loss value: 7.684643E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 1050 | lm loss value: 9.133303E+00 | lm loss PPL: 9.258550E+03 | reg loss value: 7.687837E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 1100 | lm loss value: 8.498374E+00 | lm loss PPL: 4.906784E+03 | reg loss value: 7.691198E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 1150 | lm loss value: 7.457586E+00 | lm loss PPL: 1.732959E+03 | reg loss value: 7.695348E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 1200 | lm loss value: 6.801736E+00 | lm loss PPL: 8.994072E+02 | reg loss value: 7.699996E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 1250 | lm loss value: 5.956599E+00 | lm loss PPL: 3.862942E+02 | reg loss value: 7.703359E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 1300 | lm loss value: 5.419857E+00 | lm loss PPL: 2.258468E+02 | reg loss value: 7.709359E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 1350 | lm loss value: 4.884523E+00 | lm loss PPL: 1.322274E+02 | reg loss value: 7.715837E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 1400 | lm loss value: 4.382247E+00 | lm loss PPL: 8.001767E+01 | reg loss value: 7.720004E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 1450 | lm loss value: 4.046807E+00 | lm loss PPL: 5.721447E+01 | reg loss value: 7.724000E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 1500 | lm loss value: 3.781154E+00 | lm loss PPL: 4.386665E+01 | reg loss value: 7.726649E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 1550 | lm loss value: 3.609316E+00 | lm loss PPL: 3.694079E+01 | reg loss value: 7.730050E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 1600 | lm loss value: 3.475836E+00 | lm loss PPL: 3.232483E+01 | reg loss value: 7.733351E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 1650 | lm loss value: 3.377321E+00 | lm loss PPL: 2.929220E+01 | reg loss value: 7.737789E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 1700 | lm loss value: 3.294059E+00 | lm loss PPL: 2.695204E+01 | reg loss value: 7.740013E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 1750 | lm loss value: 3.246460E+00 | lm loss PPL: 2.569920E+01 | reg loss value: 7.746004E+03 | reg loss PPL: 4.851652E+08 | 
validation loss at iteration 1800 | lm loss value: 3.218766E+00 | lm loss PPL: 2.499727E+01 | reg loss value: 7.747167E+03 | reg loss PPL: 4.851652E+08 | 
validation loss at iteration 1850 | lm loss value: 3.187777E+00 | lm loss PPL: 2.423448E+01 | reg loss value: 7.750212E+03 | reg loss PPL: 4.851652E+08 | 
validation loss at iteration 1900 | lm loss value: 3.175204E+00 | lm loss PPL: 2.393169E+01 | reg loss value: 7.753789E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 1950 | lm loss value: 3.184855E+00 | lm loss PPL: 2.416377E+01 | reg loss value: 7.757159E+03 | reg loss PPL: 4.851652E+08 | 
validation loss at iteration 2000 | lm loss value: 3.187674E+00 | lm loss PPL: 2.423201E+01 | reg loss value: 7.759353E+03 | reg loss PPL: 4.851652E+08 |
```