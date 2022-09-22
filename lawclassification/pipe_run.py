import dataset.pipe_create as pipe
import dataset.pipe_graph as graph
import dataset.fast_pipe_graph as fastGraph
import dataset.fast_pipe_graph_multi as fastGraphMulti

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
    #pipe.ohsumed_create(test_split=0.2,max_classes=23,max_row=1000000)
    #pipe.tj(test_split=0.2,max_classes=23,max_row=20000,selected_col='grupo')
    #graph.create_graph(path='twitter_chines',maxRows=1000000,windowSize=20)
    #graph.create_graph(path='r8_chines',maxRows=1000000,windowSize=20)
    #graph.create_graph(path='ohsumed',maxRows=1000000,windowSize=20)
    #graph.create_graph(path='ag_chines',maxRows=1000000,windowSize=20)
    #graph.create_graph(path='ohsumed',maxRows=1000000,windowSize=20)
    #graph.create_graph(path='scotus_lexbench',maxRows=1000000,windowSize=20)
    #graph.create_graph(path='twitter_chines',maxRows=1000000,windowSize=20)
    #pipe.tj(test_split=0.2,max_classes=10,max_rows=1000000,selected_col='cu')

    #fastGraph.fast_pipe_graph(path='twitter_chines',maxRows=1000000,windowSize=20,nThreads=16)
    #fastGraph.fast_pipe_graph(path='r8_chines',maxRows=1000000,windowSize=20,nThreads=16)
    #fastGraph.fast_pipe_graph(path='ohsumed',maxRows=1000000,windowSize=20,nThreads=16)

    fastGraph.fast_pipe_graph(path='r8_chines',maxRows=1000000,windowSize=20,nThreads=16)
    #fastGraphMulti.fast_pipe_graph(path='SemEval2018-Task1-all-data',maxRows=1000000,windowSize=20,nThreads=16)
    #fastGraphMulti.fast_pipe_graph(path='SemEval2018-Task1-all-data',maxRows=1000000,windowSize=10,nThreads=16)

    #fastGraphMulti.fast_pipe_graph(path='ecthr_a_lexbench',maxRows=1000000,windowSize=20,nThreads=16) #ok, 
    #fastGraphMulti.fast_pipe_graph(path='ecthr_b_lexbench',maxRows=1000000,windowSize=20,nThreads=16) #ok
    #fastGraphMulti.fast_pipe_graph(path='ledgar_lexbench',maxRows=1000000,windowSize=20,nThreads=16) #ok bug UNK nao encontrado ?
    #fastGraphMulti.fast_pipe_graph(path='eurlex_lexbench',maxRows=1000000,windowSize=20,nThreads=16) #ok
    #fastGraphMulti.fast_pipe_graph(path='scotus_lexbench',maxRows=1000000,windowSize=20,nThreads=16) #ok
    #fastGraphMulti.fast_pipe_graph(path='unfair_lexbench',maxRows=1000000,windowSize=20,nThreads=16) #ok

if __name__ == '__main__':
    main()