from utils.helper_funs import read_experiments
from deep_train import deep_train

def main():

    experimentsDic = read_experiments()

    for idx,experiment in enumerate(experimentsDic):

        print(f'begin of experiment: {idx+1}/{len(experimentsDic)}')
        train = deep_train(experiment)
        train.fit_and_eval()

    return None

if __name__ == '__main__':
    main()