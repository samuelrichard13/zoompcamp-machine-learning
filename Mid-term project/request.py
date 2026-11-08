import requests

url = 'http://localhost:9696/predict'


sleeper = {'user_id': 2718,
 'age': 36,
 'gender': 'Non-Binary',
 'chronotype': 'Intermediate',
 'occupation': 'Professional',
 'bedtime_hour': 0.09,
 'screen_time_before_bed_min': 66,
 'primary_app_category': 'Streaming',
 'screen_brightness_pct': 67,
 'blue_light_filter_on': 0,
 'caffeine_intake_mg': 41,
 'caffeine_last_hour': 19.2,
 'physical_activity_min': 1,
 'room_temperature_f': 75.7,
 'ambient_noise_db': 36.5,
 'sleep_onset_latency_min': 31.4,
 'total_sleep_hours': 6.14,
 'rem_sleep_pct': 17.4,
 'deep_sleep_pct': 16.5,
 'awakenings_count': 4,
 'sleep_efficiency_pct': 77.8}

response = requests.post(url, json=sleeper).json()

print(response)
