from dataset.pipe_create import yelp_review,dmoz_cats,echr_build

def main(): 
    #yelp_review(max_data_size=100000,test_split=0.2)
    #dmoz_cats(test_split=0.2)
    echr_build(problem_column='VIOLATED_ARTICLES')

if __name__ == '__main__':
    main()