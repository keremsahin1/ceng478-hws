#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include "mpi.h"
#include "mkl_cblas.h"

void parseInputs(int iArgCnt, char* sArrArgs[]);
void initData();
void initCommunicators();
double getLehmerValue(int iNumber1, int iNumber2);
void printVector(double* dpVector, int iLength);
void printMatrix(double* dpMatrix, int iRowCnt, int iColCnt);

int GiVectorLength = 20000, GiIterationCnt = 1;
int GiProcessRank = 0, GiProcessCnt = 0, GiSqrtProcCnt = 0;
double *GdArrSubMatrix, *GdArrTotalResult, *GdArrSubVector, *GdArrSubResult, *GdArrSubTotalResult;
int GiRowColCntForOneProc = 0, GiRowNoOffsetOfProc = 0, GiColNoOffsetOfProc = 0;

MPI_Group *GarrColumnGroups;
MPI_Comm *GarrColumnComms;
MPI_Group *GarrRowGroups;
MPI_Comm *GarrRowComms;

int main(int iArgCnt, char* sArrArgs[])
{
	int iIterationNo = 0, iCommNo = 0;
	double dNormOfResult = 0;
	double dTime0 = 0, dTime1 = 0, dTimeDiff = 0, dMinTimeDiff = DBL_MAX, dMaxTimeDiff = 0;

	parseInputs(iArgCnt, sArrArgs);

	MPI_Init(&iArgCnt, &sArrArgs);
	MPI_Comm_size(MPI_COMM_WORLD, &GiProcessCnt);
	MPI_Comm_rank(MPI_COMM_WORLD, &GiProcessRank);
	GiSqrtProcCnt = (int)sqrt((double)GiProcessCnt);

	initData();
	initCommunicators();

	for(iIterationNo = 0; iIterationNo < GiIterationCnt; iIterationNo++)
	{
		MPI_Barrier(MPI_COMM_WORLD);
		dTime0 = MPI_Wtime();

		for(iCommNo = 0; iCommNo < GiSqrtProcCnt; iCommNo++)
		{
			if((GiProcessRank % GiSqrtProcCnt) == iCommNo)
			{
				MPI_Barrier(GarrColumnComms[iCommNo]);
				MPI_Bcast(GdArrSubVector, GiRowColCntForOneProc, MPI_DOUBLE, 0, GarrColumnComms[iCommNo]);
				/*printf("INFO: Process%d[Comm%d] finished broadcast.\n", GiProcessRank, iCommNo);
				printVector(GdArrSubVector, GiRowColCntForOneProc);*/
			}
		}

		MPI_Barrier(MPI_COMM_WORLD);
		cblas_dgemv(CblasRowMajor, CblasNoTrans, GiRowColCntForOneProc, GiRowColCntForOneProc, 1.0, GdArrSubMatrix, GiRowColCntForOneProc, GdArrSubVector, 1, 0.0, GdArrSubResult, 1);
		/*printf("INFO: Process%d finished matrix vector multiplication.\n", GiProcessRank);
		printVector(GdArrSubResult, GiRowColCntForOneProc);*/

		for(iCommNo = 0; iCommNo < GiSqrtProcCnt; iCommNo++)
		{
			if((GiProcessRank / GiSqrtProcCnt) == iCommNo)
			{
				MPI_Barrier(GarrRowComms[iCommNo]);
				MPI_Reduce(GdArrSubResult, GdArrSubTotalResult, GiRowColCntForOneProc, MPI_DOUBLE, MPI_SUM, 0, GarrRowComms[iCommNo]);
				/*printf("INFO: Process%d finished reduce.\n", GiProcessRank);
				printVector(GdArrSubTotalResult, GiRowColCntForOneProc);*/
			}
		}

		if((GiProcessRank % GiSqrtProcCnt) == 0)
		{
			MPI_Barrier(GarrColumnComms[0]);
			MPI_Gather(GdArrSubTotalResult, GiRowColCntForOneProc, MPI_DOUBLE, GdArrTotalResult, GiRowColCntForOneProc, MPI_DOUBLE, 0, GarrColumnComms[0]);
			/* printf("INFO: Process%d finished gather.\n", GiProcessRank); */
		}

		dTime1 = MPI_Wtime();
		dTimeDiff = (dTime1 - dTime0);

		if(dTimeDiff > dMaxTimeDiff)
			dMaxTimeDiff = dTimeDiff;
		if(dTimeDiff < dMinTimeDiff)
			dMinTimeDiff = dTimeDiff;
	}

	if(GiProcessRank == 0)
	{
		/*printVector(GdArrTotalResult, GiVectorLength);*/
		dNormOfResult = cblas_dnrm2(GiVectorLength, GdArrTotalResult, 1);
		printf("Result=%f\nMin Time=%f uSec\nMax Time=%f uSec\n", dNormOfResult, (1.e6 * dMinTimeDiff), (1.e6 * dMaxTimeDiff));
	}

	MPI_Finalize();

	return 0;
}

void printVector(double* dpVector, int iLength)
{
	int iRowNo = 0;

	printf("Vector: [");

	for(iRowNo = 0; iRowNo < iLength; iRowNo++)
	{
		printf("%f, ", dpVector[iRowNo]);
	}

	printf("]\n");
}

void printMatrix(double* dpMatrix, int iRowCnt, int iColCnt)
{
	int iRowNo = 0;
	int iColNo = 0;

	printf("Matrix:\n");

	for(iRowNo = 0; iRowNo < iRowCnt; iRowNo++)
	{
		for(iColNo = 0; iColNo < iColCnt; iColNo++)
		{
			printf("%f, ", dpMatrix[(iRowNo * iColCnt) + iColNo]);
		}

		printf("\n");
	}

	printf("\n");
}

void parseInputs(int iArgCnt, char* sArrArgs[])
{
	int iArgNo = 0;

	for (iArgNo = 1; iArgNo < iArgCnt; iArgNo++)
	{
		if(strcmp("-s", sArrArgs[iArgNo]) == 0)
			GiVectorLength = atoi(sArrArgs[++iArgNo]);
		else if(strcmp("-i", sArrArgs[iArgNo]) == 0)
			GiIterationCnt = atoi(sArrArgs[++iArgNo]);
	}
}

void initData()
{
	const int ciSizeOfDouble = sizeof(double);
	int iRowNo = 0, iColNo = 0;

	GiRowColCntForOneProc = GiVectorLength / GiSqrtProcCnt;
	GiColNoOffsetOfProc = (GiProcessRank % GiSqrtProcCnt) * GiRowColCntForOneProc;
	GiRowNoOffsetOfProc = (GiProcessRank / GiSqrtProcCnt) * GiRowColCntForOneProc;

	/* Initialize matrix */
	GdArrSubMatrix = (double*)calloc((GiRowColCntForOneProc * GiRowColCntForOneProc), ciSizeOfDouble);
	for(iRowNo = 0; iRowNo < GiRowColCntForOneProc; iRowNo++)
	{
		for(iColNo = 0; iColNo < GiRowColCntForOneProc; iColNo++)
		{
			GdArrSubMatrix[(iRowNo * GiRowColCntForOneProc) + iColNo] = getLehmerValue((iRowNo + 1 + GiRowNoOffsetOfProc), (iColNo + 1 + GiColNoOffsetOfProc));
		}
	}

	/* Initialize vector */
	GdArrSubVector = (double*)calloc(GiRowColCntForOneProc, ciSizeOfDouble);
	if(GiProcessRank < GiSqrtProcCnt)
	{
		for(iRowNo = 0; iRowNo < GiRowColCntForOneProc; iRowNo++)
		{
			GdArrSubVector[iRowNo] = (double)(iRowNo + 1 + GiColNoOffsetOfProc);
		}
	}

	/* Initialize results */
	GdArrSubResult = (double*)calloc(GiRowColCntForOneProc, ciSizeOfDouble);
	GdArrSubTotalResult = (double*)calloc(GiRowColCntForOneProc, ciSizeOfDouble);
	GdArrTotalResult = (double*)calloc(GiVectorLength, ciSizeOfDouble);
}

void initCommunicators()
{
	int iColumnCommNo = 0, iRowCommNo = 0, iProcNo = 0;
	MPI_Group worldGroup;
	int iArrRanksInGroup[GiSqrtProcCnt];

	MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);

	/* Initialize column communicators */
	GarrColumnGroups = (MPI_Group*)calloc(GiSqrtProcCnt, sizeof(MPI_Group));
	GarrColumnComms = (MPI_Comm*)calloc(GiSqrtProcCnt, sizeof(MPI_Comm));
	for(iColumnCommNo = 0; iColumnCommNo < GiSqrtProcCnt; iColumnCommNo++)
	{
		for(iProcNo = 0; iProcNo < GiSqrtProcCnt; iProcNo++)
		{
			iArrRanksInGroup[iProcNo] = (iProcNo * GiSqrtProcCnt) + iColumnCommNo;
		}

		MPI_Group_incl(worldGroup, GiSqrtProcCnt, iArrRanksInGroup, &GarrColumnGroups[iColumnCommNo]);
		MPI_Comm_create(MPI_COMM_WORLD, GarrColumnGroups[iColumnCommNo], &GarrColumnComms[iColumnCommNo]);
	}

	/* Initialize row communicators */
	GarrRowGroups = (MPI_Group*)calloc(GiSqrtProcCnt, sizeof(MPI_Group));
	GarrRowComms = (MPI_Comm*)calloc(GiSqrtProcCnt, sizeof(MPI_Comm));
	for(iRowCommNo = 0; iRowCommNo < GiSqrtProcCnt; iRowCommNo++)
	{
		for(iProcNo = 0; iProcNo < GiSqrtProcCnt; iProcNo++)
		{
			iArrRanksInGroup[iProcNo] = (iRowCommNo * GiSqrtProcCnt) + iProcNo;
		}

		MPI_Group_incl(worldGroup, GiSqrtProcCnt, iArrRanksInGroup, &GarrRowGroups[iRowCommNo]);
		MPI_Comm_create(MPI_COMM_WORLD, GarrRowGroups[iRowCommNo], &GarrRowComms[iRowCommNo]);
	}
}

double getLehmerValue(int iNumber1, int iNumber2)
{
	int iMin = 0, iMax = 0;

	if(iNumber1 > iNumber2)
	{
		iMin = iNumber2;
		iMax = iNumber1;
	}
	else
	{
		iMin = iNumber1;
		iMax = iNumber2;
	}

	return (((double)iMin) / iMax);
}
