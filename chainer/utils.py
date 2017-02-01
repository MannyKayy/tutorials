from __future__ import division
import numpy as npy
## Computational Graph Visualisation
import chainer.computational_graph as c
import pydot_ng as pd
from IPython.display import Image, display

## Draw graph
def draw_cg(model):
    ''' Draws the computational graph for model '''
    graph =  pd.graph_from_dot_data(c.build_computational_graph((model,)).dump())
    return display(Image(graph.create_png()))


## Iterator for the sine function with some additional methods through the property decorator
from chainer.dataset import Iterator


class SineIteratorEven(Iterator):
    ''' Batch iterator that selects offsets evenly spaced into the dataset '''
    def __init__(self, dataset, batch_size, repeat=True):
        self.dataset = dataset
        self.batch_size = dataset.shape[0] // batch_size
        self.repeat = repeat
        
        # True if the epoch is incremented at the last iteration.
        self.is_new_epoch = False
        
        self.epoch = 0
        # Number of calls to __next__
        self.iteration = 0
        
        # Offset indices into the dataset
        self.dataset_length = dataset.shape[0]
        self.nb_batches = self.dataset_length // self.batch_size 
        self.offsets = [batch * self.batch_size for batch in range(self.nb_batches)]
        
    def __next__(self):
        if not self.repeat and self.epoch > 0:
            raise StopIteration
            
        # Iterator
        # Return item in each batch at some specific position
        x_itr = [item + self.iteration for item in self.offsets]
        x = (self.dataset[a % self.dataset_length] for a in x_itr)
        
        self.iteration += 1
        
        # Return target (x+1) for each item x
        t_itr = [item + self.iteration for item in self.offsets]
        t = (self.dataset[b % self.dataset_length] for b in t_itr)
        
        # Current epoch as a whole number
        epoch = self.iteration * self.batch_size // self.dataset_length
        
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch: self.epoch = epoch
        
        return list(zip(x,t))
    
    @property
    def epoch_detail(self):
        return self.iteration * self.batch_size / self.dataset_length
    
    def serialize(self, serializer):
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)

        
class SineIteratorSeq(Iterator):
    ''' Batch iterator that sequentially selects batches from the dataset '''
    def __init__(self, dataset, batch_size, repeat=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.repeat = repeat
        
        # True if the epoch is incremented at the last iteration.
        self.is_new_epoch = False
        
        self.epoch = 0
        # Number of calls to __next__
        self.iteration = 0
        
        # Batch offset indices into the dataset
        self.dataset_length = dataset.shape[0]
        self.nb_batches = self.dataset_length // batchsize 
        self.offsets = [batch * self.batch_size for batch in range(self.nb_batches)]
        
    def __next__(self):
        if not self.repeat and self.epoch > 0:
            raise StopIteration
            
        # Iterator
        # Return x batch
        x_itr = npy.arange(self.offsets[self.iteration % self.nb_batches],
                           self.offsets[self.iteration % self.nb_batches] 
                           + self.batch_size)
        
        #x_itr = npy.random.permutation(x_itr) # Randomize batch
        
        x = self.dataset[x_itr % self.dataset_length]
        
        
        # return target batch
        t = self.dataset[(x_itr + 1) % self.dataset_length]        
        
        
        self.iteration += 1
        
        # Current epoch as a whole number
        epoch = self.iteration * self.batch_size // self.dataset_length
        
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch: self.epoch = epoch
        
        return list(zip(x,t))
    
    @property
    def epoch_detail(self):
        return self.iteration * self.batch_size / self.dataset_length
    
    def serialize(self, serializer):
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)

### Sine Sequence Iterators
class SineIteratorEvenSequences(Iterator):
    ''' Batch sequence iterator that selects offsets evenly spaced into the dataset '''
    def __init__(self, dataset, sequence_length = 10, batch_size = 100, repeat=True):
        self.dataset = dataset
        self.dataset_max_index = self.dataset.shape[0]
        self.sequence_length = sequence_length
        self.dataset_length = (dataset.shape[0] // sequence_length) 
        self.batch_size = self.dataset_length // batch_size
        self.repeat = repeat
        
        # True if the epoch is incremented at the last iteration.
        self.is_new_epoch = False
        
        self.epoch = 0
        # Number of calls to __next__
        self.iteration = 0
        
        # Batch offset indices into the dataset
        self.nb_batches = self.dataset_length // self.batch_size
        self.offsets = [batch * self.batch_size for batch in range(self.nb_batches)]
        
    def __next__(self):
        if not self.repeat and self.epoch > 0:
            raise StopIteration

        # Iterator
        # Return x batch
        x_itr = [item + self.iteration for item in self.offsets]
        x = (self.dataset[
                ((seq * self.sequence_length) % self.dataset_max_index):
                (((seq * self.sequence_length) % self.dataset_max_index ) + self.sequence_length)]
             for seq in x_itr)
        
        self.iteration += 1
        
        t_itr = [item + self.iteration for item in self.offsets]
        t = (self.dataset[
                ((seq * self.sequence_length) % self.dataset_max_index):
                (((seq * self.sequence_length) % self.dataset_max_index) + self.sequence_length) ]
             for seq in t_itr)
        
        # Current epoch as a whole number
        epoch = (self.iteration * self.nb_batches) // self.dataset_length
        
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch: self.epoch = epoch
        
        return list(zip(x,t))
    
    @property
    def epoch_detail(self):
        return self.iteration * self.nb_batches / self.dataset_length
    
    def serialize(self, serializer):
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)


class  SineIteratorSeqSequences(Iterator):
    
    ''' Batch sequence iterator that sequentially selects batches from the dataset '''
        
    def __init__(self, dataset, sequence_length = 10, batch_size = 100, repeat=True):
        self.dataset = dataset
        self.dataset_max_index = self.dataset.shape[0]
        self.sequence_length = sequence_length # 10
        self.dataset_length = (dataset.shape[0] // sequence_length)
        self.batch_size = self.dataset_length // batch_size
        self.repeat = repeat
        
        # True if the epoch is incremented at the last iteration.
        self.is_new_epoch = False
        
        self.epoch = 0
        # Number of calls to __next__
        self.iteration = 0
        self.counter = 0
        
        # Batch offset indices into the dataset
        self.nb_batches = self.dataset_length // self.batch_size
        self.offsets = [batch * self.batch_size for batch in range(self.nb_batches)]
        
    def __next__(self):
        if not self.repeat and self.epoch > 0:
            raise StopIteration

        # Iterator
        # Return x batch
        x_itr = [item + self.iteration for item in self.offsets]
        x = []
        t = []
        for offset in self.offsets:
            curr_index = (offset  * self.sequence_length) + (self.counter * self.sequence_length)
            
            if((curr_index + (sequence_length * 2)) > self.dataset_max_index):
                curr_index = 0
                
            sample = self.dataset[curr_index : curr_index + self.sequence_length]
            x.append(sample)
            
            curr_index += self.sequence_length
            sample = self.dataset[curr_index : curr_index + self.sequence_length]
            t.append(sample)            
            
        
        self.counter += 1
        
        max_index = (self.offsets[-1]  * self.sequence_length) + (self.counter * (self.sequence_length ))
        if(max_index > self.dataset_max_index):
            self.counter = 0
            
        self.iteration += 1
        
        # Current epoch as a whole number
        epoch = (self.iteration * self.nb_batches) // self.dataset_length
        
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch: self.epoch = epoch
        
        return list(zip(x,t))
    
    @property
    def epoch_detail(self):
        return self.iteration * self.nb_batches / self.dataset_length
    
    def serialize(self, serializer):
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)
        
        

### Custom Sine Updater
from chainer import training, dataset
from chainer.training import extensions

class BPTTUpdater(training.StandardUpdater):
    ''' Updater for RNNs '''
    def __init__(self, train_iter, optimizer, bprop_len, device):
        super(BPTTUpdater, self).__init__(
            train_iter, optimizer, device=device)
        self.bprop_len = bprop_len
    
    def update_core(self):
        loss = 0
        
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')
        model = optimizer.target
        
        for i in range(self.bprop_len):
            batch = train_iter.next()
            x, t = dataset.concat_examples(batch)
            x = npy.asarray(x, npy.float32).reshape(len(x),1)
            t = npy.asarray(t, npy.float32).reshape(len(t),1)
            loss += model(Variable(x), Variable(t))

        model.cleargrads()
        loss.backward()
        loss.unchain_backward()
        optimizer.update()

        
class BPTTUpdater_BRNN(training.StandardUpdater):
    ''' Updater for the Bi-directional RNNs '''
    def __init__(self, train_iter, optimizer, bprop_len, device):
        super(BPTTUpdater_BRNN, self).__init__(
            train_iter, optimizer, device=device)
        self.bprop_len = bprop_len
    
    def update_core(self):

        loss = 0     

        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')
        model = optimizer.target      

        for i in range(self.bprop_len):         
            batch = train_iter.next()
            x, t = dataset.concat_examples(batch)
            loss += model(x, t)

        model.cleargrads()
        loss.backward()
        loss.unchain_backward()
        optimizer.update()    