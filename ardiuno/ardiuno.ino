#include <Servo.h>

Servo servo;

void setup() {
  Serial.begin(115200);
  servo.attach(9); 
  servo.write(90); 
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');

    if (command == "GOOD") {
      servo.write(0); 
      delay(1000);    
      servo.write(90);
    } else if (command == "BAD") {
      servo.write(180);
      delay(1000);     
      servo.write(90); 
    }
  }
}