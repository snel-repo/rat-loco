
//https://www.instructables.com/Arduino-Timer-Interrupts/

const byte inputPin = 2; // start recording pin
const byte intanPin = 11; // intan output
const byte outputPin = 12; // camera output

const byte modePin = A0; // mode select
const byte ledPin = LED_BUILTIN; // led pin

int mode = 0; // could be used to set different frequencies

#define MODE_0 0
#define MODE_1 1

int record_on = 0;
volatile boolean toggle1 = 0;

void setup() {
  // put your setup code here, to run once:

  pinMode(inputPin, INPUT);
  pinMode(modePin, INPUT_PULLUP);
  pinMode(outputPin, OUTPUT);
  pinMode(ledPin, OUTPUT);
  
  cli();

  //set timer1 interrupt at desired freq
  TCCR1A = 0; // set entire TCCR1A register to 0
  TCCR1B = 0; // same for TCCR1B
  TCNT1  = 0; //initialize counter value to 0
  
  /////////////////////////////////////////////////////////////////
  // vvv SECTION TO CHOOSE DESIRED FREQUENCY vvv
  // set compare match register for 125hz increments
  OCR1A = 7999; // Equation is: (16*10^6) / (2*freq*8) - 1  // must be <65536
  
  // set compare match register for 160hz increments
  // OCR1A = 6249; // Equation is: (16*10^6) / (2*freq*8) - 1  // must be <65536

  // set compare match register for 200hz increments
  // OCR1A = 4999; // Equation is: (16*10^6) / (2*freq*8) - 1  // must be <65536
  
  // set compare match register for 250hz increments
  // OCR1A = 3999; // Equation is: (16*10^6) / (2*freq*8) - 1  // must be <65536

  // set compare match register for 320hz increments
  // OCR1A = 3124; // Equation is: (16*10^6) / (2*freq*8) - 1  // must be <65536

  // set compare match register for 400hz increments
  // OCR1A = 2499; // Equation is: (16*10^6) / (2*freq*8) - 1  // must be <65536

  // set compare match register for 500hz increments
  // OCR1A = 1999; // Equation is: (16*10^6) / (2*freq*8) - 1  // must be <65536
  /////////////////////////////////////////////////////////////////
  // ^^^ SECTION TO CHOOSE DESIRED FREQUENCY ^^^
  
  // turn on CTC mode
  TCCR1B |= (1 << WGM12);
  // Set CS11 bits for 8 prescaler
  TCCR1B |= (1 << CS11);  
  // enable timer compare interrupt
  TIMSK1 |= (1 << OCIE1A);

  attachInterrupt(digitalPinToInterrupt(inputPin), start_timer, RISING);

  sei(); //allow interrupts

  //Serial.begin(38400);
}

ISR(TIMER1_COMPA_vect){ 
  //generates pulse wave of timer freq/2 (takes two cycles for full wave- toggle high then toggle low)
  if (record_on)
  {
    if (toggle1)
    {
      digitalWrite(outputPin,HIGH);
      toggle1 = 0;
    }
    else
    {
      digitalWrite(outputPin,LOW);
      toggle1 = 1;
    }
  }
  else
  {
    digitalWrite(outputPin,LOW);
  }

}


void start_timer()
{
    // reset timer count
    TCNT1 = 0;
    // set timer and intan output to HIGH
    digitalWrite(outputPin,HIGH);
    digitalWrite(intanPin,HIGH);

    // timer output starts at HIGH, so next value must be LOW (0) 
    toggle1 = 0;
}


void loop() {

  mode = 1-digitalRead(modePin);
  digitalWrite(ledPin, mode);

  record_on = digitalRead(inputPin);
  digitalWrite(intanPin, record_on);
  
}
