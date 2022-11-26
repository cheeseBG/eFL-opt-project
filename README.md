# eFL-opt-project

This project is based on [Federated-Learning-PyTorch](https://github.com/AshwinRJ/Federated-Learning-PyTorch) code.

# FL Type
### Traditional FL
- Run MLP with dirty dataset</p>
`python tfl.py --gpu=0 --dirty=70`
  
- Run CNN with dirty dataset</p>
`python tfl.py --gpu=0 --dirty=70 --model=cnn --local_ep=15 --num_users=10`
  
### eFL  
- Run MLP with dirty dataset</p>
`python efl.py --gpu=0 --dirty=70`

- Run CNN with dirty dataset</p>
`python efl.py --gpu=0 --dirty=70 --model=cnn --local_ep=15 --num_users=10`
  

# Key Features
1. LAN<->WAN: 15 times smaller than LAN bandwidth within a site , and in the worst case is 60 times
smaller-> LAN bandwidth가 더 크기때문에 communication time saving
   - deploying mobile edge nodes to collect local updates by using their computing resources in the LAN
   

2. Model parameters
   - MLP
      - num of end device: 20
      - local epoch: 87
      - local minibatch size: 32
   - MNISTCNN
      - num of end divice: 10
      - local epoch: 15
   
3. Dirty label proportion: 0.7

4. Threshold setting
    - MLP: 0.98
    - CNN: 0.9999

5. Communication Time
    - end device가 직접적으로 서버와 통신하는 경우 update time = 0.5s
