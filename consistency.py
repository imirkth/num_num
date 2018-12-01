import numpy as np
import pandas as pd

def check_consistency(model, x_valid, y_valid, eras_validate):

    eras = eras_validate.drop_duplicates()
    #print(len(eras_validate))
    #print(x_valid[1:5], y_valid[1:5])
    # eras = valid_data.era.unique()
    count = 0
    count_consistent = 0
    for era in eras:
        count += 1
        # current_valid_data = valid_data[validation_data.era == era]
        # features = [f for f in list(complete_training_data) if "feature" in f]
        loss = model.evaluate(x_valid[x_valid.era == era].drop('era',axis=1).values, y_valid[y_valid.era == era].drop('era',axis=1).values, batch_size=1024, verbose=0)[0]
        if (loss < -np.log(.5)):
            consistent = True
            count_consistent += 1
        else:
            consistent = False
        #print("{}: loss - {} consistent: {}".format(era, loss, consistent))
    print("Consistency: {}".format(count_consistent / count))
    return count_consistent / count