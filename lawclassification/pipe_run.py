import dataset.pipe_create as pipe

def main(): 
    #pipe.yelp_review(max_data_size=100000,test_split=0.2)
    #pipe.dmoz_cats(test_split=0.2)
    #pipe.echr_build(problem_column='VIOLATED_ARTICLES',max_classes = 10)
    #pipe.colab_torch_tweets()
    pipe.eurlex_lexbench(max_classes=100)
    pipe.unfair_tos_lexbench()
    pipe.ecthr_a_lexbench()
    pipe.ecthr_b_lexbench()
    pipe.scotus_lexbench()
    pipe.ledgar_lexbench()
    

if __name__ == '__main__':
    main()