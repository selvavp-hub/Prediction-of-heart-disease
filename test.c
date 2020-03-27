#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include <string.h>

#define NUMPAT 128
#define NUMIN  22
#define NUMHID 22
#define NUMOUT 1

#define rando() ((double)rand()/(RAND_MAX)+1)

int main()
{
    int count=0;
    int i, j, k, p, np, op, ranpat[NUMPAT+1], epoch;
    int NumPattern = NUMPAT, NumInput = NUMIN, NumHidden = NUMHID, NumOutput = NUMOUT;
    float Input[NUMPAT+1][NUMIN+1];
    float Test[NUMPAT+1][NUMOUT+1];
    float WeightIH[NUMIN+1][NUMHID+1];
    float WeightHO[NUMHID+1][NUMOUT+1];
    float accuracy=0, avg=0;
    
    float SumH[NUMPAT+1][NUMHID+1], Hidden[NUMPAT+1][NUMHID+1];
    float SumO[NUMPAT+1][NUMOUT+1], Output[NUMPAT+1][NUMOUT+1];

     /*reading from csv files and storing it in 2D array*/
     char IHbuffer[1024] ;
     char HObuffer[1024] ;
     char clvbuffer[1024] ;
     char testbuffer[1024] ;
          
     char *IHrecord,*IHline, *HOrecord, *HOline, *clvrecord, *clvline, *testrecord, *testline;
     int x=0,y=0;
     int l=0,m=0;
     int a=0,b=0;
     int r=0,s=0;
     FILE *IHstream = fopen("WeightsIH.csv","r");
     FILE *HOstream = fopen("WeightsHO.csv","r");
     FILE *clvstream = fopen("test.csv","r");
     FILE *teststream = fopen("result.csv","r");

     if(IHstream == NULL || HOstream == NULL || clvstream == NULL || teststream == NULL)
     {
       printf("\n file opening failed ");
       return -1 ;
     }

     while(((IHline=fgets(IHbuffer,sizeof(IHbuffer),IHstream))!=NULL && (HOline=fgets(HObuffer,sizeof(HObuffer),HOstream))!=NULL))
     {
       y=0; m=0;
       
       IHrecord = strtok(IHline,",");
       while(IHrecord != NULL)
       {
        //here you can put the record into the array as per your requirement.
        WeightIH[x][y++] = atof(IHrecord) ;
        IHrecord = strtok(NULL,",");
       }
       
       HOrecord = strtok(HOline,",");
       while (HOrecord != NULL)
       { 
        WeightHO[l][m++]= atof(HOrecord);
        HOrecord = strtok(NULL,",");       
       }
       printf("l=%d\n",l);       
       ++l ;
       ++x ; 
       
     }
     
     while((clvline=fgets(clvbuffer,sizeof(clvbuffer),clvstream))!=NULL && (testline=fgets(testbuffer,sizeof(testbuffer),teststream))!=NULL)
     {
       b=0; s=0;
       printf("Entered\n");
       clvrecord = strtok(clvline,",");
       while (clvrecord != NULL)
       { 
        Input[a][b++]= atof(clvrecord);
        clvrecord = strtok(NULL,",");       
       }       

       testrecord = strtok(testline,",");
       while (testrecord != NULL)
       { 
        Test[r][s++]= atof(testrecord);
        testrecord = strtok(NULL,",");       
       }
       printf("%d\n",a);
       ++a ;
       ++r ;
     }


	for (i=0;i<NUMIN+1;i++)
	{
		for (j=0;j<NUMHID+1;j++)
		{
			printf ("WeightsIH[%d][%d] = %f\n", i,j, WeightIH[i][j]);
		}
		printf ("\n");
	}

 
	for (i=0;i<NUMHID+1;i++)
	{
		for (j=0;j<NUMOUT+1;j++)
		{
			printf ("WeightsHO[%d][%d] = %f\n", i,j, WeightHO[i][j]);
		}
		printf ("\n");
	}

	for (i=0;i<NUMPAT+1;i++)
	{
		for (j=0;j<NUMIN+1;j++)
		{
			printf ("Input[%d][%d] = %f\n", i,j, Input[i][j]);
		}
		printf ("\n");
	}

	for (i=0;i<NUMPAT+1;i++)
	{
		for (j=0;j<NUMOUT+1;j++)
		{
			printf ("TEST[%d][%d] = %f\n", i,j, Test[i][j]);
		}
		printf ("\n");
	}

        printf ("\n");
        printf ("\n");


    fclose(IHstream);
    fclose(HOstream);
    fclose(clvstream);
    fclose(teststream);
    
    for( np = 1 ; np <= NumPattern ; np++ ) {    /* repeat for all the training patterns */

            for( j = 1 ; j <= NumHidden ; j++ ) {    /* compute hidden unit activations */


                SumH[np][j] = WeightIH[0][j] ;
                for( i = 1 ; i <= NumInput ; i++ ) {
                    SumH[np][j] += Input[np][i] * WeightIH[i][j] ;
                    //printf("SumH=%f\n",SumH[np][j]);
                }
               // printf("SumH=%f\n",SumH[np][j]);
                Hidden[np][j] = 1.0/(1.0 + exp(-SumH[np][j])) ;
            }
            
            for( k = 1 ; k <= NumOutput ; k++ ) {    /* compute output unit activations and errors */
                SumO[np][k] = WeightHO[0][k] ;
                for( j = 1 ; j <= NumHidden ; j++ ) {
                    SumO[np][k] += Hidden[np][j] * WeightHO[j][k] ;
                }
               // printf("\nSumO=%f\n",SumO[np][k]);
                //printf("SumO=%f\n",SumO[np][k]);
                Output[np][k] = 1.0/(1.0 + exp(-SumO[np][k])) ;   /* Sigmoidal Outputs */
/*              Output[p][k] = SumO[p][k];      Linear Outputs */
            }
            //printf("Test=%f\n",Test[np][1]);
            //printf("Output=%f\n\n",Output[np][1]);
            
            
            if (Test[np][1]==1 && Output[np][1]<0.5)
            {
            	printf("np (<0.5) = %d\n",np+1);
            	count++;
            }
            if (Test[np][1]==0 && Output[np][1]>0.5)
            {
            	printf ("np (>0.5) = %d\n",np+1);
            	count++;
            }
            
            
            if(Test[np][1]==0)
            accuracy+=(1-Output[np][1])/1;
            else
            accuracy+=Output[np][1]/Test[np][1];
            //printf("Accuracy=%f\n",Output[np][1]/Test[np][1]);
            
        }
        printf("\nCount = %d\n",count);
        
      /* print network outputs */
    /*for( i = 1 ; i <= NumInput ; i++ ) {
        fprintf(stdout, "Input%-4d\t", i) ;
    } */
    for( k = 1 ; k <= NumOutput ; k++ ) {
        fprintf(stdout, "Target%-4d\tOutput%-4d\t", k, k) ;
    }
    for( p = 1 ; p <= NumPattern ; p++ ) {
    fprintf(stdout, "\n%d\t", p) ;
      /*  for( i = 1 ; i <= NumInput ; i++ ) {
            fprintf(stdout, "%f\t", Input[p][i]) ;
        }*/
        for( k = 1 ; k <= NumOutput ; k++ ) {
            fprintf(stdout, "%f\t%f\t", Test[p][k], Output[p][k]) ;
        }
    }
    
    avg=(accuracy/NumPattern)*100;
    printf("\n\nAverage accuracy=%f\n",avg);
    
    fprintf(stdout, "\n\nGoodbye!\n\n") ;

}   
    
