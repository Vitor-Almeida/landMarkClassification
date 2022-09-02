import dataset.pipe_create as pipe
import dataset.pipe_graph as graph

def main(): 
    #pipe.yelp_review(max_data_size=20000,test_split=0.2)
    #pipe.dmoz_cats(test_split=0.2)
    #pipe.echr_build(problem_column='VIOLATED_ARTICLES',max_classes = 10)
    #pipe.colab_torch_tweets()
    #pipe.eurlex_lexbench(max_classes=100)
    #pipe.unfair_tos_lexbench()
    #pipe.ecthr_a_lexbench()
    #pipe.ecthr_b_lexbench()
    #pipe.scotus_lexbench()
    #pipe.ledgar_lexbench()
    #pipe.fix_scrape_landmarks(max_classes=8,test_split=0.2)
    #pipe.summary_facebook()
    #pipe.customer_complain_check_boost(max_row=10000,test_split=0.25)
    #pipe.ohsumed_create(test_split=0.2,max_classes=23,max_row=20000)
    graph.create_graph(path='ohsumed',maxRows=100000,windowSize=20)

    
if __name__ == '__main__':
    main()