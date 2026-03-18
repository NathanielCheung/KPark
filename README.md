# Kingston Smart Parking (QHacks Meter)

Real-time parking availability in downtown Kingston.

## Project Overview

This application displays parking availability for streets and lots in downtown Kingston, Ontario, with an interactive map and live status updates.

## How to run locally

Requirements: Node.js & npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating)

```sh
# Clone the repository
git clone <YOUR_GIT_URL>
cd KPark

# Install dependencies
npm i

# Start the development server
npm run dev
```

The app runs at http://localhost:5173 (or the port Vite prints).

### With hardware (ESP32 + sensors)

To run the **website with live sensor data** (Beamish Munro lot + Clergy St W ultrasonic):

1. **Start the API**: `cd server && npm install && npm start` (keeps running).
2. **Set API URL**: create `.env` in the project root with `VITE_PARKING_API_URL=http://localhost:3001` (use your PC’s local IP if the ESP32 is on another machine).
3. **Start the app**: `npm run dev` and open the URL in the browser.
4. **Flash the ESP32**: open `hardware/parking_sensor_esp32/parking_sensor_esp32.ino`, set `WIFI_SSID`, `WIFI_PASSWORD`, and `API_URL` (e.g. `http://YOUR_PC_IP:3001/api/parking`), then upload.

Full steps and wiring: see [docs/HARDWARE_SENSOR_SETUP.md](docs/HARDWARE_SENSOR_SETUP.md).  
**Once the device is plugged in:** [docs/PLUG_IN_AND_SEND_TO_SITE.md](docs/PLUG_IN_AND_SEND_TO_SITE.md) (WiFi flow + optional USB serial bridge).

## How to edit this code

**Use your preferred IDE**

Clone the repo and push changes to sync with your remote repository.

**Edit directly in GitHub**

- Navigate to the desired file(s)
- Click the "Edit" button (pencil icon) at the top right
- Make your changes and commit

**Use GitHub Codespaces**

- Navigate to the main page of your repository
- Click the "Code" button (green) near the top right
- Select the "Codespaces" tab
- Click "New codespace" to launch an environment
- Edit files directly and commit/push when done

## Technologies

- Vite
- TypeScript
- React
- shadcn-ui
- Tailwind CSS
- Leaflet (maps)
- React Query

## Deployment

Build for production:

```sh
npm run build
```

Deploy the `dist` folder to any static hosting service (Vercel, Netlify, GitHub Pages, etc.).

## Custom domain

To use a custom domain with your deployment, configure it in your hosting provider's settings (e.g., Project > Settings > Domains).
