"""
Implementation of Net2Net (http://arxiv.org/abs/1511.05641)
Numpy modules for Net2Net
- Net2Wider
- Net2Deeper
Written by Kyunghyun Paeng
https://github.com/paengs/Net2Net/blob/master/net2net.py
"""

# I added some comments for better reading

import numpy as np

class Net2Net_class(object):
    def __init__(self, error=1e-4):
        self._error_th=error
        print 'Net2Net module initialized.'
        
    def Net2Wider(self, weight1, bias1, weight2, new_width, verification=True):
        """ Net2Wider operation
         
        All weights & biases should be 'numpy' array.
        If it is 'conv' type, weight.ndim = 4 (kH, kW, InChannel, OutChannel)
        If it is 'fc' type, weight.ndim = 2 (In, Out)
        
        Args:    
            weight1: weight matrix of a target layer
            bias1: biases of a target layer, bias1.ndim = 1
            weight2: weight matrix of a next layer
            new_width: It should be larger than old width.
                     (i.e., 'conv': weight1.OutChannel < new_width,
                            'fc'  : weight1.Out < new_width )
        Returns:
            Transformed weights & biases (w1, b1, w2)
        """
        # check dimension
        assert bias1.squeeze().ndim == 1, 'Check bias.ndim'
        assert weight1.ndim == 4 or weight1.ndim == 2, 'Check weight1.ndim'
        assert weight2.ndim == 4 or weight2.ndim == 2, 'Check weight2.ndim'
        bias1 = bias1.squeeze()
        
        if weight1.ndim == 2:
            print ('inputs: FC w/b')
            assert weight1.shape[1] == weight2.shape[0], 'Check if shape of weight1 & 2 match'
            assert weight1.shape[1] == len(bias1), 'Check if shape of weight 1 & bias 1 match'
            assert weight1.shape[1] < new_width, 'new_width should be larger than old width'
            return self._wider_fc(weight1, bias1, weight2, new_width, verification)
        
        else:
            print ('inputs: conv w/b')
            assert weight1.shape[3] == weight2.shape[2], 'Check shape of weight'
            assert weight1.shape[3] == len(bias1), 'Check shape of bias'
            assert weight1.shape[3] < new_width, 'new_width should be larger than old width'
            return self._wider_conv(weight1, bias1, weight2, new_width, verification)
        
    def _wider_conv(self, teacher_w1, teacher_b1, teacher_w2, new_width, verification):
        # randint(upperbound, size), add more 3-d kernels
        rand = np.random.randint(teacher_w1.shape[3], size=(new_width-teacher_w1.shape[3]))
        # count number of replicated columns
        replication_factor = np.bincount(rand)
        student_w1 = teacher_w1.copy()
        student_w2 = teacher_w2.copy()
        student_b1 = teacher_b1.copy()
        # target layer (L) update
        for i in xrange(len(rand)):
            teacher_index = rand[i]
            new_weight = teacher_w1[:,:,:,teacher_index]
            new_weight = new_weight[:,:,:,np.newaxis]
            student_w1 = np.concatenate((student_w1, new_weight), axis=3)
            student_b1 = np.append(student_b1, teacher_b1[teacher_index])
            
        # next layer (L+1) update
        for i in xrange(len(rand)):
            teacher_index = rand[i]
            factor = replication_factor[teacher_index] + 1
            assert factor > 1, 'number of duplicated 3-d kernels in w1 should > 1, but turned out <= 1'
            new_weight = teacher_w2[:,:,teacher_index,:] * (1./factor)
            new_weight_re = new_weight[:,:,np.newaxis,:]
            student_w2 = np.concatenate((student_w2, new_weight_re), axis=2)
            #change original 3rd dim of kernel to scaled
            student_w2[:,:,teacher_index,:] = new_weight
        
        # verification neglected
        
        return student_w1, student_b1, student_w2
            
    def _wider_fc(self, teacher_w1, teacher_b1, teacher_w2, new_width, verification):
        """
         this statement implements random mapping function
         g: {1, 2, ... , q} -> {1, 2, ... , n}, where q > n
         g(j) = {j, if j < n} {random sample from {1,2,...n} if j > n}
         use g to randomly pick column of teacher weights and bias
        """
        # randint(upperbound, size)
        rand = np.random.randint(teacher_w1.shape[1], size=(new_width-teacher_w1.shape[1]))
        replication_factor = np.bincount(rand)
        student_w1 = teacher_w1.copy()
        student_w2 = teacher_w2.copy()
        student_b1 = teacher_b1.copy()
        # target layer (L) update
        for i in xrange(len(rand)):
            teacher_index = rand[i]
            # copy correspond column in teacher weight, result in a row vector
            new_weight = teacher_w1[:, teacher_index]
            # transpose to column, shape [n,] to [n,1]
            new_weight = new_weight[:, np.newaxis]
            # append to student weight
            student_w1 = np.concatenate((student_w1, new_weight), axis=1)
            # pick and append bias
            student_b1 = np.append(student_b1, teacher_b1[teacher_index])
        # next layer (L+1) update
        for i in xrange(len(rand)):
            teacher_index = rand[i]
            # total number of appearance for that specific column
            factor = replication_factor[teacher_index] + 1
            assert factor > 1, 'number of duplicated column in w1 should > 1, but turned out <= 1'
            # pick and append row
            new_weight = teacher_w2[teacher_index,:]*(1./factor)
            new_weight = new_weight[np.newaxis, :]
            student_w2 = np.concatenate((student_w2, new_weight), axis=0)
            # change original row to scaled row
            student_w2[teacher_index,:] = new_weight
        if verification:
            inputs = np.random.rand(1, teacher_w1.shape[0])
            ori1 = np.dot(inputs, teacher_w1) + teacher_b1
            ori2 = np.dot(ori1, teacher_w2)
            new1 = np.dot(inputs, student_w1) + student_b1
            new2 = np.dot(new1, student_w2)
            err = np.abs(np.sum(ori2-new2))
            assert err < self._error_th, 'Verification failed: [ERROR] {}'.format(err)
        return student_w1, student_b1, student_w2
    
    def Net2Deeper(self, weight, verification=True):
        """ Net2Deeper operation
          
        All weights & biases should be 'numpy' array.
        If it is 'conv' type, weight.ndim = 4 (kH, kW, InChannel, OutChannel)
        If it is 'fc' type, weight.ndim = 2 (In, Out)
        Args:
            weight: weight matrix where the layer to be deepened
        Returns:
            Identity matrix & bias fitted to input weight
        """
        assert weight.ndim == 4 or weight.ndim == 2, 'Check weight.ndim'
        # fc layer
        if weight.ndim == 2:
            deeper_w = np.eye(weight.shape[1]) # identical matrix
            deeper_b = np.zeros(weight.shape[1])
            if verification:
                err = np.abs(np.sum(np.dot(weight, deeper_w)-weight))
                assert err < 1e-5, 'Verification failed: [ERROR] {}'.format(err)
        # conv layer
        else:
            deeper_w = np.zeros((weight.shape[0], weight.shape[1], weight.shape[3], weight.shape[3]))
            assert weight.shape[0] % 2 == 1 and weight.shape[1] % 2 == 1, 'Kernel size [0],[1] should be odd'
            center_h = (weight.shape[0]-1)/2
            center_w = (weight.shape[1]-1)/2
            for i in range(weight.shape[3]):
                tmp = np.zeros((weight.shape[0], weight.shape[1], weight.shape[3]))
                tmp[center_h, center_w, i] = 1
                deeper_w[:,:,:,i] = tmp
            # replace the above for loop with
            #deeper_w[center_h, center_w, :, :] = 1
            deeper_b = np.zeros(weight.shape[3])
        
        # neglect verification
        #if verification:
        return deeper_w, deeper_b

if __name__ == '__main__':
    """ Net2Net Class Test """
    obj = Net2Net_class()

    w1 = np.random.rand(100, 50)
    obj.Net2Deeper(w1)
    print 'Succeed: Net2Deeper (fc)'
    
    w1 = np.random.rand(3,3,16,32)
    obj.Net2Deeper(w1)
    print 'Succeed: Net2Deeper (conv)'
    
    w1 = np.random.rand(100, 50)
    b1 = np.random.rand(50,1)
    w2 = np.random.rand(50, 10)
    obj.Net2Wider(w1, b1, w2, 70)
    print 'Succeed: Net2Wider (fc)'

    w1 = np.random.rand(3,3,16,32)
    b1 = np.random.rand(32)
    w2 = np.random.rand(3,3,32,64)
    obj.Net2Wider(w1, b1, w2, 48)
    print 'Succeed: Net2Wider (conv)'
