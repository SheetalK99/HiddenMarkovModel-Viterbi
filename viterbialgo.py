
#Programmatically implement the Viterbi algorithm to compute the most likely
#tag sequence and probability for any given observation sequence. 
#Author:Sheetal Kadam (sak170006)
import numpy as np





transition_probability = np.array([[0.3777, 0.0110, 0.0009, 0.0084, 0.0584, 0.0090, 0.0025], 
			  [0.0008,0.0002,0.7968,0.0005,0.0008,0.1698,0.0041], 
			  [0.0322, 0.0005, 0.0050, 0.0837,0.0615, 0.0514,0.2231],
			  [0.0366, 0.0004, 0.0001, 0.0733, 0.4509, 0.0036,0.0036], 
			  [0.0096,0.0176, 0.0014, 0.0086, 0.1216, 0.0177, 0.0068], 
			  [0.0068, 0.0102, 0.1011, 0.1012, 0.0120, 0.0728, 0.0479],
			  [0.1147, 0.0021, 0.0002, 0.2157, 0.4744, 0.0102, 0.0017]
			  
			  
			  
			  
			  
			  ])
pi = np.array([0.2767, 0.0006, 0.0031, 0.0453, 0.0449, 0.0510, 0.2026])

observation_likelihood = np.array([[0.000032, 0, 0, 0.000048, 0], 
			  [0, 0.308431, 0, 0, 0], 
			  [0, 0.000028, 0.000672, 0, 0.000028],
			  [0, 0, 0.000340, 0, 0], 
			  [0, 0.000200, 0.000223, 0, 0.002337], 
			  [0, 0, 0.010446, 0, 0],
			  [0, 0, 0, 0.506099, 0]])

NNP, MD, VB, JJ, NN, RB, DT = 0, 1, 2,3,4,5,6
observations = Janet,will,back,the,bill = 0, 1, 2,3,4

state_names=['NNP', 'MD', 'VB', 'JJ', 'NN', 'RB', 'DT']


def viterbi(params, observations):
    pi, A, O = params
    T = len(observations)
    S = pi.shape[0]
    
    viterbi = np.zeros((T, S))
    viterbi[:,:] = float('-inf')
    backpointers = np.zeros((T, S), 'int')
    
    # base case
    viterbi[0, :] = pi * O[:,observations[0]]
    
    # recursive case
    for t in range(1, T):
        for s2 in range(S):
            for s1 in range(S):
                score = viterbi[t-1, s1] * A[s1, s2] * O[s2, observations[t]]
                if score > viterbi[t, s2]:
                    viterbi[t, s2] = score
                    backpointers[t, s2] = s1
    
    # now follow backpointers to resolve the state sequence
    ss = []
    ss.append(np.argmax(viterbi[T-1,:]))
    for i in range(T-1, 0, -1):
        ss.append(backpointers[i, ss[-1]])
      
    ans=[state_names[i] for i in list(reversed(ss))]
    return ans, np.max(viterbi[T-1,:])
	
	
if __name__ == '__main__':
    inp1=[Janet,will,back,the,bill]
    inp2=[will,Janet,back,the,bill]
    inp3=[back,the,bill,Janet,will]
    print("Janet will back the bill:",viterbi((pi, transition_probability, observation_likelihood),inp1 ))
    print("will Janet back the bill:",viterbi((pi, transition_probability, observation_likelihood),inp2 ))
    print("back the bill Janet will:",viterbi((pi, transition_probability, observation_likelihood), inp3))

