import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
from file_operations import file_methods

class KMeansClustering:
    """
            This class shall  be used to divide the data into clusters before training.

            Written By: iNeuron Intelligence
            Version: 1.0
            Revisions: None

            """

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def elbow_plot(self,data):
        """
                        Method Name: elbow_plot
                        Description: This method saves the plot to decide the optimum number of clusters to the file.
                        Output: A picture saved to the directory
                        On Failure: Raise Exception

                        Written By: iNeuron Intelligence
                        Version: 1.0
                        Revisions: None

                """
        self.logger_object.log(self.file_object, 'Entered the elbow_plot method of the KMeansClustering class')

        wcss=[] # initializing an empty list

        try:
            for i in range (1,11):
                kmeans=KMeans(n_clusters=i,init='k-means++')    # initializing the KMeans object
                kmeans.fit(data)                                # fitting the data to the KMeans Algorithm
                wcss.append(kmeans.inertia_)
            plt.plot(range(1,11),wcss)                          # creating the graph between WCSS and the number of clusters
            plt.title('The Elbow Method')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS')
            plt.savefig('preprocessing_data/K-Means_Elbow.PNG') # saving the elbow plot locally
            # finding the value of the optimum cluster programmatically
            kn = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
            self.logger_object.log(self.file_object, 'The optimum number of clusters is: '+str(kn.knee)+' . Exited the elbow_plot method of the KMeansClustering class')
            return kn.knee

        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in elbow_plot method of the KMeansClustering class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Finding the number of clusters failed. Exited the elbow_plot method of the KMeansClustering class')
            raise Exception()

    def create_clusters(self,data,number_of_clusters):
        """
                                Method Name: create_clusters
                                Description: Create a new dataframe consisting of the cluster information.
                                Output: A datframe with cluster column
                                On Failure: Raise Exception

                                Written By: iNeuron Intelligence
                                Version: 1.0
                                Revisions: None

                        """
                        
        self.logger_object.log(self.file_object, 'Entered the create_clusters method of the KMeansClustering class')

        try:
            kmeans = KMeans(n_clusters=number_of_clusters, init='k-means++', random_state=42)
            y_kmeans=kmeans.fit_predict(data) #  divide data into clusters

            file_op = file_methods.File_Operation(self.file_object,self.logger_object)
            save_model = file_op.save_model(kmeans, 'KMeans') # saving the KMeans model to directory
            data['Cluster']=y_kmeans                          # create a new column in dataset for storing the cluster information
            self.logger_object.log(self.file_object, 'succesfully created '+str(number_of_clusters)+ 'clusters. Exited the create_clusters method of the KMeansClustering class')
            return data

        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in create_clusters method of the KMeansClustering class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Fitting the data to clusters failed. Exited the create_clusters method of the KMeansClustering class')
            raise Exception()
