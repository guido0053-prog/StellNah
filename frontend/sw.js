// StellNah Service Worker (minimal)
// Cachet die wichtigsten Dateien, damit die App "installierbar" ist.
const CACHE = "stellnah-v1";
const ASSETS = [
  "./",
  "./index.html",
  "./manifest.webmanifest",
  "./favicon.png",
  "./icons/stellnah-192.png",
  "./icons/stellnah-512.png"
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE).then((cache) => cache.addAll(ASSETS)).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  // App-Dateien aus Cache, Backend-Aufrufe nicht cachen.
  event.respondWith(
    caches.match(req).then((cached) => cached || fetch(req))
  );
});
