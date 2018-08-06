try:
    import cPickle as pickle

except:
    import pickle

y = pickle.load(open('network_siemens_data_failcount_lr0.05_as100_n1000_eps0.2_hist4_None_agent.p', 'rb'))
x = pickle.load(open('network_siemens_data_failcount_lr0.05_as100_n1000_eps0.2_hist4_None_stats.p', 'rb'))
z = pickle.load(open('network_siemens_data_failcount_lr0.05_as100_n1000_eps0.2_hist4_None_val.p', 'rb'))

print(len(x))
print(z)

print(x['detected'])
print(x['comparison']['heur_sort']['detected'])
