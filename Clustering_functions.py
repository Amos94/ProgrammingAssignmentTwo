from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans


# ### Clustering with python
# Here are some functions which can help you to print and visualize your clusters with python. We consider two algorithms: hierarchical clustering and k-means. Feel free to modify the functions and play with the default parameters.
# Here are also some good tutorials for clustering with python: 
# 
# http://brandonrose.org/clustering
# 
# http://blog.nextgenetics.net/?e=44
# 
# If you tuckle word similarity rather then word clustering or word sense disambiguation, then you would you could use the follwong source to check how well your vectors perform on word simiparity sets:
# 
# http://wordvectors.org/demo.php
# 

#######################################################################
# Print dendrogram for hierarchical clusters
#######################################################################

def hierarchical_clusters_draw(feature_matrix,target_words,max_d=0.5,
                               output_filename=None, 
                               figheight =50, figwidth=20):
    """2d plot of 'feature_matrix' using hierarhical_clusters,
       with the words labeled by 'target_words'.
    
    Parameters
    ----------    
    feature_matrix : 2d np.array,
    shape=(n_samples, n_features)
    (In our case n_samples is the target words)
    (Feature matrix is not the same as similarity/distance matrix!)
    
    target_words : list of str
        Names of the target words.
    
    max_d: float (default: 0.5)
        A threshold to apply when forming flat clusters. 
        (Play around with this parameter.)
        
    output_filename : str (default: None)
        If not None, then the output image is written to this location. 
        The filename suffix determines the image type. If None, then 
        'plt.show()' is called.
        
    figheight : int (default: 40)
        Height in display units of the output.
            
    figwidth : int (default: 20)
        Width in display units of the output.
    """
    
    Z_spat = linkage(feature_matrix, 'complete','cosine')
    # You can try out with different linkage function here:
    # 'single','complete','average'

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_figheight(50)
    fig.set_figwidth(20)
    plt.title('Hierarchical Clustering Dendrogram', fontsize=20)
    plt.ylabel('Target words', fontsize=18)
    plt.xlabel('Distance', fontsize=18)
    dendrogram(
        Z_spat,
        leaf_font_size=17.,
        labels=target_words,
        orientation='right'
    )

    plt.axvline(x = max_d, color='k', linestyle='--') 
    # This drows a vertical number, 
    # choose the value where you think clusters should be cut
    
    if output_filename:
        plt.savefig(output_filename, bbox_inches='tight') 
        #bbox_inches='tight' - to make margins minimal
    else:
        plt.show()
        

#######################################################################
# Print hierarchical clusters
#######################################################################

def hierarchical_clusters_print(feature_matrix,target_words,max_d=0.5):
    """Prints clusters in 'feature_matrix' for 'target_words'
       using hierarhical_clusters.
    
    Parameters
    ----------    
    feature_matrix : 2d np.array,
    shape=(n_samples, n_features)
    (In our case n_samples is the target words)
    (Feature matrix is not the same as similarity/distance matrix!)
    
    target_words : list of str
        Names of the target words.
    
    max_d: float (default: 0.5)
        A threshold to apply when forming flat clusters. 
        (Play around with this parameter.) 
    """

    Z_spat = linkage(feature_matrix, 'complete','cosine')
    # You can play around with different linkage function here: 
    # 'single','complete','average'
    
    clusters = fcluster(Z_spat, max_d, criterion='distance')
    num_clusters = len(set(clusters))
    
    # Printing clusters
    for ind in range(1, num_clusters+1):
        print("Cluster %d words:" % ind)

        for i,w in enumerate(target_words):
            if clusters[i] == ind:
                print(' %s' % w)
        #print #add whitespace


#######################################################################
# Print clusters with k-means algorithm
#######################################################################

#http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

def kmeans_clusters_print(feature_matrix, target_words,  
                          num_clusters = 5):
    """Prints clusters for vectors in 'feature_matrix' K-means algorithm.
    
    Parameters
    ----------    
    feature_matrix : 2d np.array, 
        shape=(n_samples, n_features)
        (In our case n_samples is the target words)
        (Feature matrix is not the same as similarity/distance matrix!)
        
    num_clusters : int (default: 5)
        Number of clusters.
        
    target_words : list of str
        Names of the target words.
    """
    
    #Fitting clusters
    km = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
    
    kmeans =  km.fit(feature_matrix)

    cluster_labels = kmeans.labels_ 
    # the array of cluster labels 
    # to which each input vector in n_samples belongs

    cluster_to_words = defaultdict(list) 
    # which word belongs to which cluster
    for c, i in enumerate(cluster_labels):
            cluster_to_words[i].append( target_words[c] )
    
    # Printing clusters
    for i in range(num_clusters):
        print("Cluster %d words:" % (i+1))
    
        for w in cluster_to_words[i]:
            print(' %s' % w)
        #print #add whitespace
    
