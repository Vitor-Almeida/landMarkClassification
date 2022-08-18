from dataset.pipe_create import yelp_review
from dataset.pipe_create import dmoz_cats

def main(): 
    yelp_review(max_data_size=10000,test_split=0.2)
    dmoz_cats(max_data_size=1000000,test_split=0.2)

if __name__ == '__main__':
    main()