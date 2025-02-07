#define TRIG_PIN 9
#define ECHO_PIN 10
#define MQ4_SENSOR A1
#define MQ135_SENSOR A0

void setup() {
  Serial.begin(9600);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
}

void loop() {
  int softness = getSoftness();
  int gasMethane = analogRead(MQ4_SENSOR);
  int gasEthylene = analogRead(MQ135_SENSOR);

  // Classify Ripeness (1-5) & Brix (5-16)
  int ripeness = classifyRipeness(softness, gasEthylene);
  float brix = calculateBrix(gasMethane, gasEthylene);
  float estimated_pH = estimatePH(ripeness, brix, gasEthylene);

  // Display Results
  Serial.print("Softness: "); Serial.print(softness);
  Serial.print(" | Ethylene: "); Serial.print(gasEthylene);
  Serial.print(" | Methane: "); Serial.print(gasMethane);
  Serial.print(" | Ripeness Level: "); Serial.print(ripeness);
  Serial.print(" | Brix (Sweetness) Level: "); Serial.print(brix, 2);
  Serial.print(" | Estimated pH: "); Serial.println(estimated_pH, 2);

  delay(2000);
}

int getSoftness() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000); 
  if (duration == 0) return 3; 

  float distance = (duration / 2.0) * 0.0343; 

  if (distance < 2.0) return 1; 
  if (distance < 2.5) return 2; 
  if (distance < 3.0) return 3; 
  if (distance < 3.5) return 4; 
  return 5; 
}

int classifyRipeness(int softness, int ethylene) {
  if (ethylene > 800 || softness == 5) return 5; // Fully Ripe
  if (ethylene > 600 || softness == 4) return 4; // Almost Ripe
  if (ethylene > 400 || softness == 3) return 3; // Mid Ripe
  if (ethylene > 250 || softness == 2) return 2; // Slightly Ripe
  return 1; // Unripe
}

float calculateBrix(int methane, int ethylene) {
  float rawBrix = 5.0; // Base sweetness level

  rawBrix += (methane / 1200.0) * 5.5; // Methane influences sugar levels
  rawBrix += (ethylene / 1200.0) * 4.0; // Ethylene affects sugar accumulation

  return constrain(rawBrix, 5.0, 16.0); // Keep Brix in range
}

float estimatePH(int ripeness, float brix, int ethylene) {
  float estimated_pH = 4.5; // Base pH

  estimated_pH -= (ripeness - 1) * 0.2; // Riper fruits have lower pH
  estimated_pH -= (brix - 5) * 0.05; // Higher sugar reduces pH
  estimated_pH -= (ethylene / 1000.0) * 0.1; // Ethylene slightly lowers pH

  return constrain(estimated_pH, 3.0, 4.5);
}