import umqtt.simple as mqtt

BROKER = "your.mqtt.server"
TOPIC = "pet/sound"

def send_mqtt():
    client = mqtt.MQTTClient("esp32", BROKER)
    client.connect()
    with open("pet_sound.wav", "rb") as f:
        audio_data = f.read()
    client.publish(TOPIC, audio_data)
    client.disconnect()
    print("MQTT Upload Done!")
