




So some notes as we figure this out

The Board (game) needs to:
- be reset (set up the conditions for a new game)
- Get state (get any sensor info to feed to robot)
- step (take an action)
    - update the robots position
    returns:
        - current state, rewared, done

The NN
- inherits from nn.Module
- establishes a nn stack with different layers
- forward()
    -  Standard pytorch method for calc'ing predictions
    - when you do nnet(x) where x is an input tensor, forward is called automatically to generate the prediction

Uhm ok so our simple example is failing unfortunately. its just getting stuck with 1 or 2 as the action...

I need a more simple example









