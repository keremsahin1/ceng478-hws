#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include "mkl_cblas.h"

void parseInputs(int iArgCnt, char* sArrArgs[]);
void initData();
double getLehmerValue(int iNumber1, int iNumber2);

int GiVectorLength = 10240, GiIterationCnt = 100;
int GiProcessRank = 0, GiProcessCnt = 0;
double *GdArrSubMatrix, *GdArrTotalResult, *GdArrVector, *GdArrSubResult;
int GiColCntForOneProc = 0, GiColOffsetOfProc = 0;

int main(int iArgCnt, char* sArrArgs[])
{
	int iIterationNo = 0;
	double dNormOfResult = 0;
	double dTime0 = 0, dTime1 = 0, dTimeDiff = 0, dMinTimeDiff = DBL_MAX, dMaxTimeDiff = 0;

	parseInputs(iArgCnt, sArrArgs);

	MPI_Init(&iArgCnt, &sArrArgs);
	MPI_Comm_size(MPI_COMM_WORLD, &GiProcessCnt);
	MPI_Comm_rank(MPI_COMM_WORLD, &GiProcessRank);

	initData();

	for(iIterationNo = 0; iIterationNo < GiIterationCnt; iIterationNo++)
	{
		MPI_Barrier(MPI_COMM_WORLD);
		dTime0 = MPI_Wtime();

		cblas_dgemv(CblasRowMajor, CblasNoTrans, GiVectorLength, GiColCntForOneProc, 1.0, GdArrSubMatrix, GiColCntForOneProc, GdArrVector, 1, 0.0, GdArrSubResult, 1);

		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Reduce(GdArrSubResult, GdArrTotalResult, GiVectorLength, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

		dTime1 = MPI_Wtime();
		dTimeDiff = (dTime1 - dTime0);

		if(dTimeDiff > dMaxTimeDiff)
			dMaxTimeDiff = dTimeDiff;
		if(dTimeDiff < dMinTimeDiff)
			dMinTimeDiff = dTimeDiff;
	}

	if(GiProcessRank == 0)
	{
		dNormOfResult = cblas_dnrm2(GiVectorLength, GdArrTotalResult, 1);
		printf("Result=%f\nMin Time=%f uSec\nMax Time=%f uSec\n", dNormOfResult, (1.e6 * dMinTimeDiff), (1.e6 * dMaxTimeDiff));
	}

	MPI_Finalize();

	return 0;
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

	GiColCntForOneProc = GiVectorLength / GiProcessCnt;
	GiColOffsetOfProc = GiProcessRank * GiColCntForOneProc;

	/* Initialize matrix */
	GdArrSubMatrix = (double*)malloc(GiColCntForOneProc * GiVectorLength * ciSizeOfDouble);
	for(iRowNo = 0; iRowNo < GiVectorLength; iRowNo++)
	{
		for(iColNo = 0; iColNo < GiColCntForOneProc; iColNo++)
		{
			GdArrSubMatrix[(iRowNo * GiColCntForOneProc) + iColNo] = getLehmerValue((iRowNo + 1), (iColNo + 1 + GiColOffsetOfProc));
		}
	}

	/* Initialize vector */
	GdArrVector = (double*)malloc(GiColCntForOneProc * ciSizeOfDouble);
	for(iRowNo = 0; iRowNo < GiColCntForOneProc; iRowNo++)
	{
		GdArrVector[iRowNo] = (double)(GiColOffsetOfProc + iRowNo + 1);
	}

	/* Initialize results */
	GdArrSubResult = (double*)malloc(GiVectorLength * ciSizeOfDouble);
	if(GiProcessRank == 0)
	{
		GdArrTotalResult = (double*)malloc(GiVectorLength * ciSizeOfDouble);
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
