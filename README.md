# NeuralNetwork
以1層 input 1層 hidden 1層 output ，以下為訓練時改變權重值之數學推倒過程</br></br>
![demo](https://github.com/Alex-CHUN-YU/NeuralNetwork/blob/master/image/demo.png)</br></br>

## 使用方式
Input:</br>
```
1.執行 neural_network.py 
2.程式中輸入的 data
輸入 data:
[0,1,1]
[0,1,1]
[1,1,1]
[1,0,1]
預期輸出 data:
[0]
[0]
[1]
[1]
```
Output:</br>
```
Error:0.472341171698
Error:0.0172567538663
Error:0.0116280592805
Error:0.00928031972489
Error:0.00792244386867
Error:0.00701366941342
Error:0.00635222626448
Error:0.00584371996314
Error:0.0054374123008
Error:0.00510331685603
Output after traning
[[ 0.00481738]
 [ 0.00481738]
 [ 0.99311339]
 [ 0.99723057]]
```

## 開發環境
Python 3.5.2</br>
pip install numpy</br>

## 致謝
WMMKS 學長</br>
Siraj Raval youtube 教學(Build a Neural Net in 4 Minutes)</br>
