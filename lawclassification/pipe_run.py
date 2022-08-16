from dataset.pipe_create import yelp_review

def main():
    yelp_review(max_data_size=1000,test_split=0.2)

if __name__ == '__main__':
    main()