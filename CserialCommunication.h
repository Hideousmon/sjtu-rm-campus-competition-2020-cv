//
// Created by USER-Z on 2020/11/9.
//

#ifndef NINEBLOCKDEMO_CSERIALCOMMUNICATION_H
#define NINEBLOCKDEMO_CSERIALCOMMUNICATION_H
#include <wiringPi.h>
#include <wiringSerial.h>

void Serial_Communication(int ans) {
    int fd;
    if (wiringPiSetup() < 0) return;
    if ((fd = serialOpen("/dev/ttyS1", 115200)) < 0)return;
    char head = '@';
    char i = ans % 7;
    serialPutchar(fd, head);
    serialPutchar(fd, i);
}
#endif //NINEBLOCKDEMO_CSERIALCOMMUNICATION_H
