// Copyright (c) 2007 Intel Corp.

// Black-Scholes
// Analytical method for calculating European options

// Reference Source: Options, Futures, and Other Derivatives
// 3rd Edition, Prentice Hall, John C. Hull


#include <cstdio>
#include <cmath>
#include <iostream>
#include <chrono>


#define DIVIDE 120.0
#define fptype float  // Precision to use for calculations
#define NUM_RUNS 1
#define PAD 256
#define LINESIZE 64


typedef struct OptionData_ {
        fptype s;          // spot price
        fptype strike;     // strike price
        fptype r;          // risk-free interest rate
        fptype divq;       // dividend rate
        fptype v;          // volatility
        fptype t;          // time to maturity or option expiration in years
                           //     (1yr = 1.0, 6mos = 0.5, 3mos = 0.25, ..., etc)
        char OptionType;   // Option type.  "P"=PUT, "C"=CALL
        fptype divs;       // dividend vals (not used in this test)
        fptype DGrefval;   // DerivaGem Reference Value
} OptionData;

OptionData* data;
fptype* prices;
int numOptions;

int*    otype;
fptype* sptprice;
fptype* strike;
fptype* rate;
fptype* volatility;
fptype* otime;
int numError = 0;


// Cumulative Normal Distribution Function
#define inv_sqrt_2xPI 0.39894228040143270286

fptype CNDF (fptype InputX) {
    int sign;
    fptype OutputX;
    fptype xInput;
    fptype xNPrimeofX;
    fptype expValues;
    fptype xK2;
    fptype xK2_2, xK2_3;
    fptype xK2_4, xK2_5;
    fptype xLocal, xLocal_1;
    fptype xLocal_2, xLocal_3;

    // Check for negative value of InputX
    if (InputX < 0.0) {
        InputX = -InputX;
        sign = 1;
    } else {
        sign = 0;
    }

    xInput = InputX;

    // Compute NPrimeX term common to both four & six decimal accuracy calcs
    expValues = exp(-0.5f * InputX * InputX);
    xNPrimeofX = expValues;
    xNPrimeofX = xNPrimeofX * inv_sqrt_2xPI;

    xK2 = 0.2316419 * xInput;
    xK2 = 1.0 + xK2;
    xK2 = 1.0 / xK2;
    xK2_2 = xK2 * xK2;
    xK2_3 = xK2_2 * xK2;
    xK2_4 = xK2_3 * xK2;
    xK2_5 = xK2_4 * xK2;

    xLocal_1 = xK2 * 0.319381530;
    xLocal_2 = xK2_2 * (-0.356563782);
    xLocal_3 = xK2_3 * 1.781477937;
    xLocal_2 = xLocal_2 + xLocal_3;
    xLocal_3 = xK2_4 * (-1.821255978);
    xLocal_2 = xLocal_2 + xLocal_3;
    xLocal_3 = xK2_5 * 1.330274429;
    xLocal_2 = xLocal_2 + xLocal_3;

    xLocal_1 = xLocal_2 + xLocal_1;
    xLocal   = xLocal_1 * xNPrimeofX;
    xLocal   = 1.0 - xLocal;
    OutputX  = xLocal;

    if (sign) {
        OutputX = 1.0 - OutputX;
    }

    return OutputX;
}


fptype BlkSchlsEqEuroNoDiv(fptype sptprice,
                           fptype strike, fptype rate, fptype volatility,
                           fptype time, int otype, float timet, fptype* N1, fptype* N2) {
    fptype OptionPrice;

    // local private working variables for the calculation
    fptype xRiskFreeRate;
    fptype xVolatility;
    fptype xTime;
    fptype xSqrtTime;

    fptype logValues;
    fptype xLogTerm;
    fptype xD1;
    fptype xD2;
    fptype xPowerTerm;
    fptype xDen;
    fptype d1;
    fptype d2;
    fptype FutureValueX;
    fptype NofXd1;
    fptype NofXd2;
    fptype NegNofXd1;
    fptype NegNofXd2;

    xRiskFreeRate = rate;
    xVolatility = volatility;
    xTime = time;

    xSqrtTime = sqrt(xTime);

    logValues = log( sptprice / strike );

    xLogTerm = logValues;

    xPowerTerm = xVolatility * xVolatility;
    xPowerTerm = xPowerTerm * 0.5;

    xD1 = xRiskFreeRate + xPowerTerm;
    xD1 = xD1 * xTime;
    xD1 = xD1 + xLogTerm;

    xDen = xVolatility * xSqrtTime;
    xD1 = xD1 / xDen;
    xD2 = xD1 -  xDen;

    d1 = xD1;
    d2 = xD2;

    NofXd1 = CNDF(d1);
    if (NofXd1 > 1.0) {
        std::cerr << "Greater than one!" << std::endl ;
    }

    NofXd2 = CNDF(d2);
    if (NofXd2 > 1.0) {
        std::cerr << "Greater than one!" << std::endl ;
    }

    *N1 = NofXd1;
    *N2 = NofXd2;

    FutureValueX = strike * exp(-rate*time);
    if (otype == 0) {
        OptionPrice = (sptprice * NofXd1) - (FutureValueX * NofXd2);
    } else {
        NegNofXd1 = (1.0 - NofXd1);
        NegNofXd2 = (1.0 - NofXd2);
        OptionPrice = (FutureValueX * NegNofXd2) - (sptprice * NegNofXd1);
    }

    return OptionPrice;
}


int bs_thread(void* tid_ptr) {
    int i, j;
    int tid = *(int*)tid_ptr;
    int start = tid * (numOptions);
    int end = start + (numOptions);
    fptype price_orig;

    for (j = 0; j < NUM_RUNS; j++) {
        for (i = start; i < end; i++) {
            /* Calling main function to calculate option value based on
             * Black & Scholes's equation.
             */
            fptype N1, N2;
            price_orig = BlkSchlsEqEuroNoDiv(sptprice[i], strike[i],
                                             rate[i], volatility[i], otime[i],
                                             otype[i], 0, &N1, &N2);
            prices[i] = price_orig;
        }
    }
    return 0;
}


int main (int argc, char* argv[]) {
    FILE *file;
    int i;
    int loopnum;
    fptype* buffer;
    int* buffer2;
    int rv;

    fflush(NULL);

    char* inputFile = argv[1];
    char* outputFile = argv[2];

    //Read input data from file
    file = fopen(inputFile, "r");
    if (file == NULL) {
        printf("ERROR: Unable to open file `%s'.\n", inputFile);
        exit(1);
    }
    rv = fscanf(file, "%i", &numOptions);
    if (rv != 1) {
        printf("ERROR: Unable to read from file `%s'.\n", inputFile);
        fclose(file);
        exit(1);
    }

    // Alloc spaces for the option data
    data = (OptionData*)malloc(numOptions * sizeof(OptionData));
    prices = (fptype*)malloc(numOptions * sizeof(fptype));
    for (loopnum = 0; loopnum < numOptions; ++ loopnum) {
        rv = fscanf(file, "%f %f %f %f %f %f %c %f %f", &data[loopnum].s, &data[loopnum].strike, &data[loopnum].r,
                                                        &data[loopnum].divq, &data[loopnum].v, &data[loopnum].t,
                                                        &data[loopnum].OptionType, &data[loopnum].divs, &data[loopnum].DGrefval);
        if (rv != 9) {
            printf("ERROR: Unable to read from file `%s'.\n", inputFile);
            fclose(file);
            exit(1);
        }
    }
    rv = fclose(file);
    if (rv != 0) {
        printf("ERROR: Unable to close file `%s'.\n", inputFile);
        exit(1);
    }

    buffer = (fptype*)malloc(5 * numOptions * sizeof(fptype) + PAD);
    sptprice = (fptype*)(((unsigned long long)buffer + PAD) & ~(LINESIZE - 1));
    strike = sptprice + numOptions;
    rate = strike + numOptions;
    volatility = rate + numOptions;
    otime = volatility + numOptions;

    buffer2 = (int*)malloc(numOptions * sizeof(fptype) + PAD);
    otype = (int*)(((unsigned long long)buffer2 + PAD) & ~(LINESIZE - 1));

    for (i = 0; i < numOptions; i++) {
        otype[i]      = (data[i].OptionType == 'P') ? 1 : 0;
        sptprice[i]   = data[i].s / DIVIDE;
        strike[i]     = data[i].strike / DIVIDE;
        rate[i]       = data[i].r;
        volatility[i] = data[i].v;
        otime[i]      = data[i].t;
    }

    // Serial version
    int tid = 0;
    auto begin = std::chrono::high_resolution_clock::now();
    bs_thread(&tid);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() << " us" << std::endl;

    // Write prices to output file
    file = fopen(outputFile, "w");
    if (file == NULL) {
        printf("ERROR: Unable to open file `%s'.\n", outputFile);
        exit(1);
    }
    if (rv < 0) {
        printf("ERROR: Unable to write to file `%s'.\n", outputFile);
        fclose(file);
        exit(1);
    }
    for (i = 0; i < numOptions; i++) {
        rv = fprintf(file, "%.18f\n", prices[i]);
        if (rv < 0) {
            printf("ERROR: Unable to write to file `%s'.\n", outputFile);
            fclose(file);
            exit(1);
        }
    }
    rv = fclose(file);
    if (rv != 0) {
        printf("ERROR: Unable to close file `%s'.\n", outputFile);
        exit(1);
    }

    free(data);
    free(prices);

    return 0;
}
