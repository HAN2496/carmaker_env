#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <wiringPi.h>
#include <softPwm.h>
#include <math.h>

#define LOOPTIME 5

#define ENC2REDGEAR 216

#define MOTOR1 19 // upper at motor driver (close to Board)
#define MOTOR2 26 // lower at motor driver
#define ENCODERA 5 // purple
#define ENCODERB 6 // blue
#define PULSEINPUT 13

float pgain = 120;
float igain = 0.1;
float dgain = 0.79;
float t = 0.01;

float g1;
float g2;
float g3;

int encA;
int encB;
int encoderPosition = 0;
float redGearPosition = 0;
float referencePosition = 10;
float errorPosition = 0;

float e = 0;
float e1 = 0;
float e2 = 0;
float m = 0;
float m1 = 0;

unsigned int checkTime;
unsigned int checkTimeBefore;

float control();
void funcEncoderA();
void funcEncoderB();
void pulse_catch();

int total_num;
int trial_nth[10] = {};
int current_trial = -1;

float total_error = 0;
int pulse_time = 0;

int main(void){
    wiringPiSetupGpio();
    pinMode(ENCODERA, INPUT);
    pinMode(ENCODERB, INPUT);
    pinMode(PULSEINPUT, INPUT);

    softPwmCreate(MOTOR1, 0, 100);
    softPwmCreate(MOTOR2, 0, 100);

    wiringPiISR(ENCODERA, INT_EDGE_BOTH, funcEncoderA);
    wiringPiISR(ENCODERB, INT_EDGE_BOTH, funcEncoderB);
    wiringPiISR(PULSEINPUT, INT_EDGE_RISING, pulse_catch);

    g1 = pgain + igain * t + dgain / t;
    g2 = -1 * (pgain + 2 * dgain / t);
    g3 = dgain / t;

    softPwmWrite(MOTOR1, 0);
    softPwmWrite(MOTOR2, 0);
    delay(1000);

    printf("Enter Total Trial Number: ");
    scanf("%d", &total_num);
    printf("\n");

    int for_scan;

    for(int i = 0; i<total_num; i++){
	printf("%d trial target :", i+1);
	scanf("%d", &for_scan);
	trial_nth[i] = for_scan;
    }


    while(1){
	if(current_trial>-1){
	    checkTime = millis();
	    if(checkTime - checkTimeBefore > LOOPTIME){
		if (errorPosition > 0){
		    softPwmWrite(MOTOR1, control());
		    softPwmWrite(MOTOR2, 0);
		} else {
		    softPwmWrite(MOTOR2, -1 * control());
		    softPwmWrite(MOTOR1, 0);
		}
		checkTimeBefore = checkTime;
		total_error += LOOPTIME * fabs(errorPosition) * (checkTime - pulse_time);
	    }

	    if(current_trial == total_num){
		softPwmWrite(MOTOR1, 0);
		softPwmWrite(MOTOR2, 0);
		printf("%.5f", total_error);
		break;
	    }
	}
    }
    return 0;
}

float control(){
    e = errorPosition;
    m = m1 + g1 * e + g2 * e1 + g3 * e2;
    e2 = e1;
    e1 = e;
    m1 = m;
    return m;
}

void funcEncoderA(){
    encA = digitalRead(ENCODERA);
    encB = digitalRead(ENCODERB);
    if(encA == HIGH){
        if(encB == LOW) encoderPosition ++;
        else encoderPosition --;
    } else{
        if(encB == LOW) encoderPosition --;
        else encoderPosition ++;
    }
    redGearPosition = (float) encoderPosition / ENC2REDGEAR;
    printf("funcEncoderA() A : %d B : %d gearPos : %f\n",
    encA, encB, redGearPosition);
    errorPosition = referencePosition - redGearPosition;
}

void funcEncoderB(){
    encA = digitalRead(ENCODERA);
    encB = digitalRead(ENCODERB);
    if(encB == HIGH){
        if(encA == LOW) encoderPosition --;
        else encoderPosition ++;
    } else{
        if(encA == LOW) encoderPosition ++;
        else encoderPosition --;
    }
    redGearPosition = (float) encoderPosition / ENC2REDGEAR;
    printf("funcEncoderB() A : %d B : %d gearPos : %f\n", encA, encB, redGearPosition);
    errorPosition = referencePosition - redGearPosition;
}

void pulse_catch(){
    if(millis() - pulse_time>500){
		printf("pulsecatch\n");
		current_trial++;

		referencePosition  = trial_nth[current_trial];
		errorPosition = referencePosition - redGearPosition;

		checkTimeBefore = millis();
		checkTime = millis();
		pulse_time = millis();
    }

}

