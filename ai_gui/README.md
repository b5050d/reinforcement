# AI GUI

## Overview
    This little application assists me in the training and management of models.

### Design Objectives
- Provide a button to train the game or play the game
- Provide a list of all the previous runs to play
- Provide a plot of the evaluation data (reward and such)
- Provide an input so that I can tag the runs as I do them
- Auto date stamp results 
- Model training output


Notes on versioning of differnet things
- As long as the observation space stays the same, then we are ok to change up the config for the game. We could upload bunches of different configs and then run the model on it. Lets stick with this for now. If we need to create more complex simulation then we create a whole new space for it. We will call the first simulation the medium level example.