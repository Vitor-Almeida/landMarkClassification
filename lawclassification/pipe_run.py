import dataset.pipe_create as pipe
import dataset.fast_pipe_graph_multi as fastGraphMulti
import dataset.append_token_hier as app

#we always need to run pipe. app. fastGraphMulti.

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
    #pipe.tj(test_split=0.2,selected_col='grupo_Hid')

    #pipe.big_tj(test_split=0.05)
    #app.append_token_hier(dataname='big_tj',modelname='rufimelo.Legal-BERTimbau-base',hier_max_seg=64,hier_max_seg_length=128)

    #pipe.big_tj_single_mixed(test_split=0.20)
    #app.append_token_hier(dataname='big_tj_single_mixed',modelname='rufimelo.Legal-BERTimbau-base',hier_max_seg=64,hier_max_seg_length=128)

    #pipe.big_tj_single(test_split=0.20)

    pipe.small_tj_single(test_split=0.20)
    app.append_token_hier(dataname='small_tj_single',modelname='rufimelo.Legal-BERTimbau-base',hier_max_seg=64,hier_max_seg_length=128)

    #pipe.big_tj_single(test_split=0.05)
    #app.append_token_hier(dataname='big_tj_single',modelname='rufimelo.Legal-BERTimbau-base',hier_max_seg=64,hier_max_seg_length=128)
    #fastGraphMulti.fast_pipe_graph(path='big_tj',maxRows=1000000,windowSize=20,nThreads=16, train=False,modelname='dominguesm.legal-bert-base-cased-ptbr')

    #pipe.small_tj(test_split=0.2)
    #app.append_token_hier(dataname='small_tj',modelname='dominguesm.legal-bert-base-cased-ptbr')
    #app.append_token_hier(dataname='small_tj',modelname='neuralmind.bert-base-portuguese-cased')
    #fastGraphMulti.fast_pipe_graph(path='small_tj',maxRows=1000000,windowSize=20,nThreads=16, train=False,modelname='dominguesm.legal-bert-base-cased-ptbr')

    #fastGraphMulti.fast_pipe_graph(path='r8_chines',maxRows=1000000,windowSize=20,nThreads=16)
    #fastGraphMulti.fast_pipe_graph(path='ohsumed',maxRows=1000000,windowSize=20,nThreads=16)

    #fastGraphMulti.fast_pipe_graph(path='r8_chines',maxRows=1000000,windowSize=20,nThreads=16)
    #fastGraphMulti.fast_pipe_graph(path='SemEval2018-Task1-all-data',maxRows=1000000,windowSize=20,nThreads=16)
    #fastGraphMulti.fast_pipe_graph(path='SemEval2018-Task1-all-data',maxRows=1000000,windowSize=10,nThreads=16)

    #fastGraphMulti.fast_pipe_graph(path='ohsumed',maxRows=1000000,windowSize=20,nThreads=16)
    #fastGraphMulti.fast_pipe_graph(path='ohsumed',maxRows=1000000,windowSize=20,nThreads=16)
    #fastGraphMulti.fast_pipe_graph(path='r8_chines',maxRows=1000000,windowSize=20,nThreads=16)

    #app.append_token_hier(dataname='ohsumed',modelname='bert-base-uncased')
    #fastGraphMulti.fast_pipe_graph(path='ohsumed',maxRows=1000000,windowSize=20,nThreads=16, train=True,modelname='bert-base-uncased')

    #app.append_token_hier(dataname='SemEval2018-Task1-all-data',modelname='bert-base-uncased')
    #fastGraphMulti.fast_pipe_graph(path='SemEval2018-Task1-all-data',maxRows=1000000,windowSize=20,nThreads=16, train=False, modelname='bert-base-uncased')

    #app.append_token_hier(dataname='scotus_lexbench',modelname='nlpaueb.legal-bert-base-uncased')
    #fastGraphMulti.fast_pipe_graph(path='scotus_lexbench',maxRows=1000000,windowSize=20,nThreads=16, train=False,modelname='nlpaueb.legal-bert-base-uncased')

    #app.append_token_hier(dataname='ledgar_lexbench',modelname='nlpaueb.legal-bert-base-uncased')
    #fastGraphMulti.fast_pipe_graph(path='ledgar_lexbench',maxRows=1000000,windowSize=20,nThreads=16, train=False,modelname='nlpaueb.legal-bert-base-uncased')

    #app.append_token_hier(dataname='ecthr_b_lexbench',modelname='nlpaueb.legal-bert-base-uncased')
    #fastGraphMulti.fast_pipe_graph(path='ecthr_b_lexbench',maxRows=1000000,windowSize=20,nThreads=16, train=False,modelname='nlpaueb.legal-bert-base-uncased')

    #app.append_token_hier(dataname='ecthr_a_lexbench',modelname='nlpaueb.legal-bert-base-uncased')
    #fastGraphMulti.fast_pipe_graph(path='ecthr_a_lexbench',maxRows=1000000,windowSize=20,nThreads=16, train=False,modelname='nlpaueb.legal-bert-base-uncased')

    #app.append_token_hier(dataname='eurlex_lexbench',modelname='nlpaueb.legal-bert-base-uncased')
    #fastGraphMulti.fast_pipe_graph(path='eurlex_lexbench',maxRows=1000000,windowSize=20,nThreads=16, train=False,modelname='nlpaueb.legal-bert-base-uncased')

    #app.append_token_hier(dataname='unfair_lexbench',modelname='nlpaueb.legal-bert-base-uncased')
    #fastGraphMulti.fast_pipe_graph(path='unfair_lexbench',maxRows=1000000,windowSize=20,nThreads=16, train=False,modelname='nlpaueb.legal-bert-base-uncased')


    #app.append_token_hier(dataname='unfair_lexbench',modelname='nlpaueb.legal-bert-base-uncased')
    #fastGraphMulti.fast_pipe_graph(path='unfair_lexbench',maxRows=1000000,windowSize=20,nThreads=16, train=False,modelname='nlpaueb.legal-bert-base-uncased')

    #app.append_token_hier(dataname='scotus_lexbench',modelname='nlpaueb.legal-bert-base-uncased')
    

    #fastGraphMulti.fast_pipe_graph(path='ledgar_lexbench',maxRows=1000000,windowSize=20,nThreads=16, train=False)
    
    #fastGraphMulti.fast_pipe_graph(path='ecthr_b_lexbench',maxRows=1000000,windowSize=20,nThreads=16, train=False)
    #fastGraphMulti.fast_pipe_graph(path='ecthr_a_lexbench',maxRows=1000000,windowSize=20,nThreads=16, train=False)
    #fastGraphMulti.fast_pipe_graph(path='scotus_lexbench',maxRows=1000000,windowSize=20,nThreads=16, train=False)
    #fastGraphMulti.fast_pipe_graph(path='eurlex_lexbench',maxRows=1000000,windowSize=20,nThreads=16, train=False)
    #fastGraphMulti.fast_pipe_graph(path='unfair_lexbench',maxRows=1000000,windowSize=20,nThreads=16, train=False)

    #fastGraphMulti.fast_pipe_graph(path='SemEval2018-Task1-all-data',maxRows=1000000,windowSize=20,nThreads=16)
    #fastGraphMulti.fast_pipe_graph(path='ohsumed',maxRows=1000000,windowSize=20,nThreads=16)
    #fastGraphMulti.fast_pipe_graph(path='tj',maxRows=1000000,windowSize=20,nThreads=16)

if __name__ == '__main__':
    main()