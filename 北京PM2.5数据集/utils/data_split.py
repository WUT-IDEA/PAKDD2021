import numpy as np

def split_sequence_parallel(train_seq,sw_width,pred_length):

    X,y=[],[]
    for i in range(len(train_seq)):
        end_index = i + sw_width
        out_end_index = end_index + pred_length
        if out_end_index > len(train_seq) :
            break

        # [i:end_index, :] 截取第i行到第end_index-1行、所有列的数据；
        # [end_index:out_end_index, :] 截取第end_index行到第out_end_index行、所有列的数据；
        x, y1 = train_seq[i:end_index, :], train_seq[end_index:out_end_index, 0]
        X.append(x)
        y.append(y1)

    X,y = np.array(X), np.array(y)
    features = X.shape[2]
    return X, y, features