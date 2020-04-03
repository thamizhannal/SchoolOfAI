

#  T3D Implementation Flow Chart



###  

### STEP 1

We initialize the Experience Replay Memory with a size of 1e6. Then we populate it with new transitions

![Step1]( https://raw.githubusercontent.com/thamizhannal/EIP3/master/p2s9_images/s1.png)

Here a sigle trasition consist of state, next state, reward and action.
s-current state
s'-next state
r-reward
a-action



# STEP 2

### Build one DNN for the Actor model and one for Actor Target

Actor: Initialized with state_dim, action_dim and max_action variable. It creates Linear layers using state_dim and action dim.

Init(): Initialize Linear layer using state & action dimension and max_action.

forward() : create feed ward network of 3 layers.

layer1: FC with relue

layer2: FC with relu

layer3: max_action * tanh(layer2)

Here max_action is to clip in case we added too much noise.



![Step2]( https://raw.githubusercontent.com/thamizhannal/EIP3/master/p2s9_images/s2.png )


## 	STEP 3

### Build two DNNs for the two Critic models and two DNNs for the two Critic Targets

Init(): Initialize two Linear layer for Critic1 & Critic2 network using state & action dimension and max_action. 

forward() : create feed ward network of 3 layers for critic1 & critic2.

layer: FC with relue

Q1():  To update Q values using the x-state, u=action arguments.



![Step3]( https://raw.githubusercontent.com/thamizhannal/EIP3/master/p2s9_images/s3.png )

### STEP 4-15

### Training process. Create a T3D class, initialize variables and get ready for step 4 .

### Building the whole training process into a class called T3D.

#### init(): 

Initialize Actor model & target Actor model.

Initialize Adam optimizer for Actor with model weights to keep them same for both models

Initialize Critic model & target Critic model.

Initialize Adam optimizer for Critic  with model weights to keep them same for both models



select_action:

Given state it chooses an action to be taken by actor model.



### T3D Algorithm Train Flow chart -Step4-Step15

<img src="https://raw.githubusercontent.com/thamizhannal/EIP3/master/p2s9_images/s4s15.png" alt="Step4-s15" style="zoom:150%;" />

### STEP 4

	Sample from a batch of transitions (s, s', a, r) from the memory



###  STEP 5

	From the next state s', the actor target plays the next action a'

### STEP 6

	We add Gaussian noise to this next action a' and we clamp it in a range of values supported by the environment

### STEP 7

	The two Critic targets take each the couple (s', a') as input and return two Q values, Qt1(s', a') and Qt2(s', a') as outputs

###  STEP 8

			Keep the minimum of these two Q-Values

### STEP 9 

			We get the final target of the two Critic models, which is:
			Qt = r + gamma * min(Qt1, Qt2)
			We can define 
			target_q or Qt as reward + discount  * torch.min(Qt1, Qt2)
			target_Q = reward +(1-done)*discount*target_Q
			0 = episode not over, 1 = episode over
			we can't run the above equation efficiently as some components are in computational graphs and some are not. we need to mane one minor modification

### STEP 10 

	Two critic models take (s, a) and return two Q-Vales

### STEP 11

	Compute the Critic Loss

### STEP 12

		Backpropagate this critic loss and update the parameters of two Critic models

### STEP 13 

​	Once every two iterations, we update our Actor model by performing gradient ASCENT on the output of the first Critic model

### STEP 14

	Still, in once every two iterations, we update our Actor Target by Polyak Averaging				

### STEP 15 

	Still, in once every two iterations, we update our  Critic Target by Polyak Averaging
