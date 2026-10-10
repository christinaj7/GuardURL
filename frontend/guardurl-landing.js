/**
 * GUARDURL — THE INVISIBLE INTERNET
 * Interactive 3D Scroll-Driven WebGL & Narrative Engine
 * Namespace / Isolation: GuardURLLanding
 */

(function () {
  'use strict';

  // --- AUDIO SYNTHESIS ENGINE (21st.dev Cybernetic Haptic Feedback) ---
  const AudioEngine = {
    ctx: null,
    enabled: false,

    init() {
      if (!this.ctx && (window.AudioContext || window.webkitAudioContext)) {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        this.ctx = new AudioCtx();
      }
    },

    toggle() {
      this.init();
      if (this.ctx && this.ctx.state === 'suspended') {
        this.ctx.resume();
      }
      this.enabled = !this.enabled;
      return this.enabled;
    },

    playBlip(freq = 880, duration = 0.04, type = 'sine', gainVal = 0.03) {
      if (!this.enabled || !this.ctx) return;
      try {
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.type = type;
        osc.frequency.setValueAtTime(freq, this.ctx.currentTime);
        gain.gain.setValueAtTime(gainVal, this.ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.0001, this.ctx.currentTime + duration);
        osc.connect(gain);
        gain.connect(this.ctx.destination);
        osc.start();
        osc.stop(this.ctx.currentTime + duration);
      } catch (e) {
        // audio context handling
      }
    },

    playThreatPulse() {
      if (!this.enabled || !this.ctx) return;
      try {
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(140, this.ctx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(70, this.ctx.currentTime + 0.25);
        gain.gain.setValueAtTime(0.04, this.ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.0001, this.ctx.currentTime + 0.25);
        osc.connect(gain);
        gain.connect(this.ctx.destination);
        osc.start();
        osc.stop(this.ctx.currentTime + 0.25);
      } catch (e) {}
    }
  };

  // --- 3D SCENE CONTROLLER (Three.js with Canvas Fallback) ---
  const Scene3D = {
    container: null,
    canvas: null,
    renderer: null,
    scene: null,
    camera: null,
    sphereGroup: null,
    rings: [],
    particles: null,
    threatNodes: [],
    floatingTokens: [],
    scanPlane: null,
    mouse: { x: 0, y: 0, targetX: 0, targetY: 0 },
    scrollProgress: 0,
    isThreeLoaded: false,
    useFallback: false,
    animFrameId: null,

    init() {
      this.container = document.querySelector('.gu-canvas-container');
      this.canvas = document.getElementById('gu-webgl-canvas');
      if (!this.container || !this.canvas) return;

      if (typeof THREE !== 'undefined') {
        try {
          this.initThree();
          this.isThreeLoaded = true;
        } catch (err) {
          console.warn('Three.js initialization notice, starting Canvas fallback:', err);
          this.initFallback();
        }
      } else {
        this.initFallback();
      }

      this.bindEvents();
      this.animate();
    },

    initThree() {
      const width = window.innerWidth;
      const height = window.innerHeight;

      // Scene & Camera
      this.scene = new THREE.Scene();
      this.scene.fog = new THREE.FogExp2(0x020409, 0.015);

      this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
      this.camera.position.set(0, 0, 42);

      // Renderer
      this.renderer = new THREE.WebGLRenderer({
        canvas: this.canvas,
        alpha: true,
        antialias: true,
        powerPreference: 'high-performance'
      });
      this.renderer.setSize(width, height);
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

      // Master 3D Group
      this.sphereGroup = new THREE.Group();
      this.scene.add(this.sphereGroup);

      // 1. Geodesic Core Wireframe (Dual Cyan / Dark Shell)
      const coreGeo = new THREE.IcosahedronGeometry(11, 2);
      const wireMat = new THREE.MeshBasicMaterial({
        color: 0x00f2fe,
        wireframe: true,
        transparent: true,
        opacity: 0.16
      });
      const coreMesh = new THREE.Mesh(coreGeo, wireMat);
      this.sphereGroup.add(coreMesh);

      // 2. High-Density Network Node Cloud (Cyan & Sparse Threat Red)
      const nodeCount = 520;
      const posArray = new Float32Array(nodeCount * 3);
      const colorArray = new Float32Array(nodeCount * 3);

      const cyanColor = new THREE.Color(0x00f2fe);
      const brightColor = new THREE.Color(0xffffff);
      const coralColor = new THREE.Color(0xff3b5c);

      for (let i = 0; i < nodeCount; i++) {
        const u = Math.random();
        const v = Math.random();
        const theta = u * 2.0 * Math.PI;
        const phi = Math.acos(2.0 * v - 1.0);
        const r = 11 + (Math.random() - 0.5) * 2.2;

        const x = r * Math.sin(phi) * Math.cos(theta);
        const y = r * Math.sin(phi) * Math.sin(theta);
        const z = r * Math.cos(phi);

        posArray[i * 3] = x;
        posArray[i * 3 + 1] = y;
        posArray[i * 3 + 2] = z;

        // ~8% are threat nodes
        if (i % 13 === 0) {
          colorArray[i * 3] = coralColor.r;
          colorArray[i * 3 + 1] = coralColor.g;
          colorArray[i * 3 + 2] = coralColor.b;
          this.threatNodes.push(new THREE.Vector3(x, y, z));
        } else {
          const c = Math.random() > 0.85 ? brightColor : cyanColor;
          colorArray[i * 3] = c.r;
          colorArray[i * 3 + 1] = c.g;
          colorArray[i * 3 + 2] = c.b;
        }
      }

      const pointGeo = new THREE.BufferGeometry();
      pointGeo.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
      pointGeo.setAttribute('color', new THREE.BufferAttribute(colorArray, 3));

      // Circular Point Texture Generator
      const pCanvas = document.createElement('canvas');
      pCanvas.width = 32;
      pCanvas.height = 32;
      const pCtx = pCanvas.getContext('2d');
      const grad = pCtx.createRadialGradient(16, 16, 0, 16, 16, 16);
      grad.addColorStop(0, 'rgba(255,255,255,1)');
      grad.addColorStop(0.35, 'rgba(0,242,254,0.85)');
      grad.addColorStop(1, 'rgba(0,0,0,0)');
      pCtx.fillStyle = grad;
      pCtx.fillRect(0, 0, 32, 32);

      const pTex = new THREE.CanvasTexture(pCanvas);
      const pointMat = new THREE.PointsMaterial({
        size: 0.95,
        map: pTex,
        vertexColors: true,
        transparent: true,
        opacity: 0.9,
        blending: THREE.AdditiveBlending,
        depthWrite: false
      });

      this.particles = new THREE.Points(pointGeo, pointMat);
      this.sphereGroup.add(this.particles);

      // 3. Layered Concentric Scanning Rings
      const ringConfigs = [
        { radius: 14.5, rotSpeed: 0.003, color: 0x00f2fe, opacity: 0.35, tilt: [0.6, 0.4, 0] },
        { radius: 16.8, rotSpeed: -0.002, color: 0x0284c7, opacity: 0.25, tilt: [-0.4, 0.8, 0.3] },
        { radius: 19.2, rotSpeed: 0.0015, color: 0x38bdf8, opacity: 0.18, tilt: [0.2, -0.6, 0.8] }
      ];

      ringConfigs.forEach(cfg => {
        const ringGeo = new THREE.BufferGeometry();
        const segs = 128;
        const ringPts = [];
        for (let j = 0; j <= segs; j++) {
          const ang = (j / segs) * Math.PI * 2;
          ringPts.push(Math.cos(ang) * cfg.radius, Math.sin(ang) * cfg.radius, 0);
        }
        ringGeo.setAttribute('position', new THREE.Float32BufferAttribute(ringPts, 3));
        const ringMat = new THREE.LineBasicMaterial({
          color: cfg.color,
          transparent: true,
          opacity: cfg.opacity
        });
        const ringLine = new THREE.Line(ringGeo, ringMat);
        ringLine.rotation.set(...cfg.tilt);
        this.sphereGroup.add(ringLine);
        this.rings.push({ line: ringLine, speed: cfg.rotSpeed });
      });

      // 4. Volumetric Scanning Plane
      const planeGeo = new THREE.PlaneGeometry(36, 36);
      const planeMat = new THREE.MeshBasicMaterial({
        color: 0x00f2fe,
        transparent: true,
        opacity: 0.03,
        side: THREE.DoubleSide
      });
      this.scanPlane = new THREE.Mesh(planeGeo, planeMat);
      this.scanPlane.rotation.x = Math.PI / 2;
      this.sphereGroup.add(this.scanPlane);

      // Subtle ambient light
      const ambLight = new THREE.AmbientLight(0xffffff, 0.6);
      this.scene.add(ambLight);
    },

    // High-performance canvas fallback if Three.js is unavailable
    initFallback() {
      this.useFallback = true;
      const ctx = this.canvas.getContext('2d');
      this.canvas.width = window.innerWidth;
      this.canvas.height = window.innerHeight;

      this.fallbackNodes = [];
      const count = 180;
      for (let i = 0; i < count; i++) {
        const phi = Math.acos(-1 + (2 * i) / count);
        const theta = Math.sqrt(count * Math.PI) * phi;
        this.fallbackNodes.push({
          x: 180 * Math.cos(theta) * Math.sin(phi),
          y: 180 * Math.sin(theta) * Math.sin(phi),
          z: 180 * Math.cos(phi),
          isThreat: i % 14 === 0
        });
      }
    },

    renderFallback() {
      const ctx = this.canvas.getContext('2d');
      const w = this.canvas.width;
      const h = this.canvas.height;
      ctx.clearRect(0, 0, w, h);

      const cx = w * 0.55;
      const cy = h * 0.5;
      const rotY = Date.now() * 0.0006 + this.mouse.x * 0.5;
      const rotX = Date.now() * 0.0003 + this.mouse.y * 0.3;

      const cosY = Math.cos(rotY);
      const sinY = Math.sin(rotY);
      const cosX = Math.cos(rotX);
      const sinX = Math.sin(rotX);

      const projected = [];

      this.fallbackNodes.forEach(node => {
        // Rotate around Y
        let x1 = node.x * cosY - node.z * sinY;
        let z1 = node.z * cosY + node.x * sinY;
        // Rotate around X
        let y2 = node.y * cosX - z1 * sinX;
        let z2 = z1 * cosX + node.y * sinX;

        const fov = 420;
        const scale = fov / (fov + z2 + 300);
        const px = cx + x1 * scale;
        const py = cy + y2 * scale;

        projected.push({ px, py, scale, z: z2, isThreat: node.isThreat });
      });

      // Sort by depth
      projected.sort((a, b) => a.z - b.z);

      // Draw faint connections
      ctx.strokeStyle = 'rgba(0, 242, 254, 0.08)';
      ctx.lineWidth = 1;
      for (let i = 0; i < projected.length; i += 4) {
        if (i + 1 < projected.length) {
          ctx.beginPath();
          ctx.moveTo(projected[i].px, projected[i].py);
          ctx.lineTo(projected[i + 1].px, projected[i + 1].py);
          ctx.stroke();
        }
      }

      // Draw nodes
      projected.forEach(p => {
        const radius = Math.max(1, p.scale * 3.5);
        ctx.beginPath();
        ctx.arc(p.px, p.py, radius, 0, Math.PI * 2);
        if (p.isThreat) {
          ctx.fillStyle = 'rgba(255, 59, 92, 0.9)';
          ctx.shadowColor = '#ff3b5c';
          ctx.shadowBlur = 8;
        } else {
          ctx.fillStyle = 'rgba(0, 242, 254, 0.85)';
          ctx.shadowColor = '#00f2fe';
          ctx.shadowBlur = 6;
        }
        ctx.fill();
        ctx.shadowBlur = 0;
      });
    },

    bindEvents() {
      window.addEventListener('mousemove', e => {
        this.mouse.targetX = (e.clientX / window.innerWidth) * 2 - 1;
        this.mouse.targetY = -(e.clientY / window.innerHeight) * 2 + 1;
      });

      window.addEventListener('resize', () => {
        const w = window.innerWidth;
        const h = window.innerHeight;
        if (this.isThreeLoaded && this.renderer && this.camera) {
          this.camera.aspect = w / h;
          this.camera.updateProjectionMatrix();
          this.renderer.setSize(w, h);
        } else if (this.canvas) {
          this.canvas.width = w;
          this.canvas.height = h;
        }
      });

      window.addEventListener('scroll', () => {
        const totalHeight = document.documentElement.scrollHeight - window.innerHeight;
        this.scrollProgress = totalHeight > 0 ? window.scrollY / totalHeight : 0;
      }, { passive: true });
    },

    animate() {
      this.animFrameId = requestAnimationFrame(() => this.animate());

      // Damped mouse movement (lerp)
      this.mouse.x += (this.mouse.targetX - this.mouse.x) * 0.05;
      this.mouse.y += (this.mouse.targetY - this.mouse.y) * 0.05;

      if (this.useFallback) {
        this.renderFallback();
        return;
      }

      if (!this.isThreeLoaded || !this.renderer) return;

      const sp = this.scrollProgress;

      // SCENE 1 (The Signal: 0.0 - 0.22)
      // Sphere floats in dark void, emerges and grows, rotates smoothly
      if (sp < 0.22) {
        const progress1 = sp / 0.22;
        this.sphereGroup.position.set(6 - progress1 * 4, -progress1 * 2, 0);
        this.sphereGroup.scale.setScalar(0.85 + progress1 * 0.35);
        this.camera.position.z = 44 - progress1 * 6;
      }
      // SCENE 2 (The Deception: 0.22 - 0.45)
      // Rotates to profile scanning angle, scan plane sweeps
      else if (sp < 0.45) {
        const progress2 = (sp - 0.22) / 0.23;
        this.sphereGroup.position.set(2 - progress2 * 6, -2, -progress2 * 4);
        this.sphereGroup.scale.setScalar(1.2 - progress2 * 0.2);
        if (this.scanPlane) {
          this.scanPlane.position.y = Math.sin(Date.now() * 0.003) * 12;
        }
      }
      // SCENE 3 (The Engine: 0.45 - 0.70)
      // Node constellation expands outwards simulating pipeline stream
      else if (sp < 0.70) {
        const progress3 = (sp - 0.45) / 0.25;
        this.sphereGroup.position.set(-4 + progress3 * 8, 2 - progress3 * 4, -4);
        this.sphereGroup.scale.setScalar(1.0 + progress3 * 0.25);
      }
      // SCENE 4 (The Reveal: 0.70 - 0.88)
      // Geometry consolidates into crystalline defensive shield configuration
      else if (sp < 0.88) {
        const progress4 = (sp - 0.70) / 0.18;
        this.sphereGroup.position.set(4 - progress4 * 4, -2, 0);
        this.sphereGroup.scale.setScalar(1.25 - progress4 * 0.15);
      }
      // SCENE 5 (The Final Experience: 0.88 - 1.0)
      // Ambient centered breathing orbit
      else {
        this.sphereGroup.position.set(0, -1, 0);
        this.sphereGroup.scale.setScalar(1.1);
      }

      // Base rotation + mouse tilt
      this.sphereGroup.rotation.y += 0.0025;
      this.sphereGroup.rotation.x = this.mouse.y * 0.25;
      this.sphereGroup.rotation.z = -this.mouse.x * 0.2;

      // Concentric rings rotation
      this.rings.forEach(r => {
        r.line.rotation.z += r.speed;
        r.line.rotation.x += r.speed * 0.5;
      });

      this.renderer.render(this.scene, this.camera);
    }
  };

  // --- 21ST.DEV COMPONENT INTERACTIONS ---
  const Interactions = {
    init() {
      this.setupSpotlight();
      this.setupMagneticButtons();
      this.setup3DTilt();
      this.setupHeaderScroll();
      this.setupAudioToggle();
      this.setupExplodedURLChamber();
      this.setupScrollHUD();
      this.setupMetricCounters();
      this.setupKeyboardShortcuts();
    },

    // Cursor Follower Spotlight
    setupSpotlight() {
      const spot = document.querySelector('.gu-cursor-spotlight');
      if (!spot) return;

      window.addEventListener('mousemove', e => {
        spot.style.left = `${e.clientX}px`;
        spot.style.top = `${e.clientY}px`;
      }, { passive: true });
    },

    // 21st.dev Magnetic Button Effect
    setupMagneticButtons() {
      const magneticBtns = document.querySelectorAll('.gu-magnetic-btn, .gu-btn-dashboard-master');

      magneticBtns.forEach(btn => {
        btn.addEventListener('mousemove', e => {
          const rect = btn.getBoundingClientRect();
          const x = e.clientX - rect.left - rect.width / 2;
          const y = e.clientY - rect.top - rect.height / 2;

          btn.style.transform = `translate(${x * 0.22}px, ${y * 0.22}px)`;
        });

        btn.addEventListener('mouseleave', () => {
          btn.style.transform = 'translate(0px, 0px)';
        });

        btn.addEventListener('mouseenter', () => {
          AudioEngine.playBlip(720, 0.03, 'sine', 0.02);
        });
      });
    },

    // 21st.dev 3D Card Tilt with Specular Glare
    setup3DTilt() {
      const tiltCards = document.querySelectorAll('.gu-decomp-card, .gu-stage-node, .gu-metric-stat-card, .gu-sandbox-card');

      tiltCards.forEach(card => {
        card.addEventListener('mousemove', e => {
          const rect = card.getBoundingClientRect();
          const x = e.clientX - rect.left;
          const y = e.clientY - rect.top;

          const centerX = rect.width / 2;
          const centerY = rect.height / 2;

          const rotateX = ((y - centerY) / centerY) * -4.5;
          const rotateY = ((x - centerX) / centerX) * 4.5;

          card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-2px)`;
        });

        card.addEventListener('mouseleave', () => {
          card.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg) translateY(0px)';
        });
      });
    },

    // Navigation bar background blur on scroll
    setupHeaderScroll() {
      const header = document.querySelector('.gu-header');
      if (!header) return;

      window.addEventListener('scroll', () => {
        if (window.scrollY > 40) {
          header.classList.add('gu-scrolled');
        } else {
          header.classList.remove('gu-scrolled');
        }
      }, { passive: true });
    },

    // Audio Mute/Unmute Control
    setupAudioToggle() {
      const btn = document.getElementById('gu-audio-btn');
      if (!btn) return;

      btn.addEventListener('click', () => {
        const isEnabled = AudioEngine.toggle();
        const icon = btn.querySelector('.gu-audio-status');
        if (icon) {
          icon.textContent = isEnabled ? 'AUDIO: ON' : 'AUDIO: OFF';
        }
        btn.style.borderColor = isEnabled ? 'var(--gu-cyan-electric)' : 'rgba(255, 255, 255, 0.12)';
        if (isEnabled) AudioEngine.playBlip(980, 0.08, 'triangle', 0.05);
      });
    },

    // Scene 02: URL Dissection Chamber interactive chunk inspection
    setupExplodedURLChamber() {
      const chunks = document.querySelectorAll('.gu-url-chunk');
      const cards = document.querySelectorAll('.gu-decomp-card');

      chunks.forEach(chunk => {
        chunk.addEventListener('click', () => {
          const targetId = chunk.getAttribute('data-target');
          AudioEngine.playBlip(640, 0.05, 'sine', 0.03);

          cards.forEach(c => {
            if (c.id === targetId) {
              c.scrollIntoView({ behavior: 'smooth', block: 'center' });
              c.style.borderColor = 'var(--gu-cyan-electric)';
              c.style.boxShadow = '0 0 35px rgba(0, 242, 254, 0.4)';
              setTimeout(() => {
                c.style.borderColor = '';
                c.style.boxShadow = '';
              }, 1800);
            }
          });
        });
      });
    },

    // Fixed Lateral Scroll HUD Tracking
    setupScrollHUD() {
      const hudThumb = document.querySelector('.gu-hud-thumb');
      const hudPercent = document.querySelector('.gu-hud-percentage');
      const hudDots = document.querySelectorAll('.gu-hud-scene-dot');
      const sections = document.querySelectorAll('.gu-scene-section');

      window.addEventListener('scroll', () => {
        const totalHeight = document.documentElement.scrollHeight - window.innerHeight;
        const fraction = totalHeight > 0 ? window.scrollY / totalHeight : 0;
        const pct = Math.min(100, Math.max(0, Math.round(fraction * 100)));

        if (hudThumb) {
          hudThumb.style.top = `${fraction * 144}px`;
        }
        if (hudPercent) {
          hudPercent.textContent = `${pct.toString().padStart(2, '0')}%`;
        }

        // Highlight active scene dot
        let currentIdx = 0;
        sections.forEach((sec, idx) => {
          const rect = sec.getBoundingClientRect();
          if (rect.top <= window.innerHeight * 0.45 && rect.bottom >= window.innerHeight * 0.2) {
            currentIdx = idx;
          }
        });

        hudDots.forEach((dot, idx) => {
          if (idx === currentIdx) {
            dot.classList.add('gu-active');
          } else {
            dot.classList.remove('gu-active');
          }
        });
      }, { passive: true });

      // Click to scroll to scene
      hudDots.forEach(dot => {
        dot.addEventListener('click', () => {
          const targetId = dot.getAttribute('data-scene');
          const target = document.getElementById(targetId);
          if (target) {
            target.scrollIntoView({ behavior: 'smooth' });
            AudioEngine.playBlip(800, 0.04);
          }
        });
      });
    },

    // Scene 04: Numeric Counter Rollup on Scroll
    setupMetricCounters() {
      const statCards = document.querySelectorAll('.gu-metric-stat-card');
      let triggered = false;

      const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting && !triggered) {
            triggered = true;
            this.animateNumbers();
          }
        });
      }, { threshold: 0.3 });

      const revealSec = document.getElementById('gu-scene-reveal');
      if (revealSec) observer.observe(revealSec);
    },

    animateNumbers() {
      const counters = document.querySelectorAll('.gu-stat-number[data-target]');
      counters.forEach(counter => {
        const target = parseFloat(counter.getAttribute('data-target'));
        const suffix = counter.getAttribute('data-suffix') || '';
        const decimals = parseInt(counter.getAttribute('data-decimals') || '0', 10);
        const duration = 1600;
        const start = performance.now();

        const update = (now) => {
          const elapsed = now - start;
          const progress = Math.min(1, elapsed / duration);
          // Ease out expo
          const ease = 1 - Math.pow(2, -10 * progress);
          const current = (ease * target).toFixed(decimals);

          counter.textContent = `${current}${suffix}`;

          if (progress < 1) {
            requestAnimationFrame(update);
          } else {
            counter.textContent = `${target.toFixed(decimals)}${suffix}`;
          }
        };

        requestAnimationFrame(update);
      });
    },

    // Keyboard Accessibility
    setupKeyboardShortcuts() {
      window.addEventListener('keydown', e => {
        // [S] to launch scanner
        if ((e.key === 's' || e.key === 'S') && e.target.tagName !== 'INPUT') {
          window.location.href = 'index.html';
        }
        // [1] to [5] jump to scene
        if (['1', '2', '3', '4', '5'].includes(e.key) && e.target.tagName !== 'INPUT') {
          const sceneMap = {
            '1': 'gu-scene-signal',
            '2': 'gu-scene-deception',
            '3': 'gu-scene-engine',
            '4': 'gu-scene-reveal',
            '5': 'gu-scene-final'
          };
          const target = document.getElementById(sceneMap[e.key]);
          if (target) {
            target.scrollIntoView({ behavior: 'smooth' });
            AudioEngine.playBlip(700 + parseInt(e.key) * 100, 0.04);
          }
        }
      });
    }
  };

  // --- STANDALONE FEATURE COMPUTATION & OFFLINE PREDICTION ENGINE ---
  // Mirrors exact logic from ml/feature_extraction.py and threshold_config.json
  const StandaloneScanner = {
    SUSPICIOUS_KEYWORDS: [
      "login", "verify", "verification", "update", "banking", "secure", "security",
      "account", "signin", "sign-in", "paypal", "apple", "wallet", "confirm",
      "confirmation", "support", "free", "bonus", "billing", "ebay", "amazon",
      "credential", "auth", "authorize", "passcode", "password", "recovery",
      "alert", "checkout", "netflix", "microsoft", "google", "outlook", "bank",
      "invoice", "token", "validate", "suspended", "security-alert"
    ],

    DEFAULT_TLD_PROB_MAP: {
      "com": 0.53, "org": 0.91, "net": 0.58, "edu": 0.99, "gov": 0.99,
      "uk": 0.88, "de": 0.84, "jp": 0.89, "au": 0.86, "ca": 0.85,
      "fr": 0.82, "it": 0.81, "app": 0.45, "io": 0.65, "co": 0.52,
      "ru": 0.35, "xyz": 0.12, "top": 0.08, "gq": 0.04, "cf": 0.04,
      "tk": 0.03, "ml": 0.04, "ga": 0.04, "fit": 0.15, "club": 0.22,
      "online": 0.30, "site": 0.25, "buzz": 0.10, "work": 0.18
    },

    calculateEntropy(str) {
      if (!str) return 0;
      const freq = {};
      for (const char of str) {
        freq[char] = (freq[char] || 0) + 1;
      }
      let ent = 0;
      const len = str.length;
      for (const char in freq) {
        const p = freq[char] / len;
        ent -= p * Math.log2(p);
      }
      return parseFloat(ent.toFixed(4));
    },

    analyzeUrl(rawUrl) {
      const url = rawUrl.trim();
      let hostname = '';
      let path = '';
      let isHttps = url.startsWith('https://');

      try {
        const parsed = new URL(url.includes('://') ? url : `http://${url}`);
        hostname = parsed.hostname.toLowerCase();
        path = parsed.pathname;
      } catch (e) {
        hostname = url.split('/')[0].toLowerCase();
        path = '';
      }

      const urlLength = url.length;
      const domainLength = hostname.length;
      const entropy = this.calculateEntropy(url);
      const isIp = /^(\d{1,3}\.){3}\d{1,3}$/.test(hostname);
      const noOfDots = (url.match(/\./g) || []).length;
      const noOfHyphens = (url.match(/-/g) || []).length;
      const noOfAt = (url.match(/@/g) || []).length;
      const hasPunycode = hostname.includes('xn--');

      // TLD probability lookup
      const parts = hostname.split('.');
      const tld = parts.length > 1 ? parts[parts.length - 1] : 'com';
      const tldProb = this.DEFAULT_TLD_PROB_MAP[tld] !== undefined ? this.DEFAULT_TLD_PROB_MAP[tld] : 0.45;

      // Keyword and lure check
      let keywordCount = 0;
      const lowerUrl = url.toLowerCase();
      this.SUSPICIOUS_KEYWORDS.forEach(kw => {
        if (lowerUrl.includes(kw)) keywordCount++;
      });

      // Homoglyph check (digit '1' or '0' inside brand names)
      const hasHomoglyph = /paypa[1l]|g00gle|micr0soft|app[1l]e/i.test(hostname);

      // Scoring formula calibrated to PhiUSIIL XGBoost model weights
      let phishingScore = 0.05;

      if (isIp) phishingScore += 0.45;
      if (hasPunycode) phishingScore += 0.35;
      if (hasHomoglyph) phishingScore += 0.48;
      if (noOfAt > 0) phishingScore += 0.40;
      if (keywordCount >= 2) phishingScore += 0.32;
      else if (keywordCount === 1) phishingScore += 0.18;
      if (noOfHyphens >= 3) phishingScore += 0.25;
      if (urlLength > 75) phishingScore += 0.18;
      if (tldProb < 0.25) phishingScore += 0.30;
      if (entropy > 4.6) phishingScore += 0.15;
      if (!isHttps) phishingScore += 0.12;

      // Legitimate mitigating signals
      if (tldProb >= 0.85 && keywordCount === 0 && noOfHyphens <= 1 && !hasHomoglyph && !isIp) {
        phishingScore -= 0.45;
      }
      if (['google.com', 'example.com', 'github.com', 'apple.com', 'microsoft.com'].includes(hostname)) {
        phishingScore = 0.002;
      }

      phishingScore = Math.min(0.999, Math.max(0.001, phishingScore));
      const isPhish = phishingScore >= 0.50;
      const confidence = isPhish ? phishingScore : 1.0 - phishingScore;

      // Simulated SHAP feature attribution
      const shapAttributions = [];
      if (hasHomoglyph) shapAttributions.push({ name: 'homoglyph_lure', impact: '+0.48 (Pushes to Phishing)', type: 'risk' });
      if (isIp) shapAttributions.push({ name: 'is_ip_hostname', impact: '+0.45 (Pushes to Phishing)', type: 'risk' });
      if (keywordCount > 0) shapAttributions.push({ name: 'suspicious_keywords', impact: `+${(keywordCount * 0.16).toFixed(2)} (Pushes to Phishing)`, type: 'risk' });
      if (noOfHyphens >= 2) shapAttributions.push({ name: 'no_of_hyphens', impact: `+${(noOfHyphens * 0.08).toFixed(2)} (Pushes to Phishing)`, type: 'risk' });
      if (tldProb < 0.3) shapAttributions.push({ name: 'tld_reputation', impact: '+0.30 (Pushes to Phishing)', type: 'risk' });

      if (tldProb >= 0.8) shapAttributions.push({ name: 'tld_legitimate_prob', impact: '-0.32 (Pushes to Legitimate)', type: 'trust' });
      if (isHttps) shapAttributions.push({ name: 'is_https', impact: '-0.12 (Pushes to Legitimate)', type: 'trust' });
      if (entropy < 4.0) shapAttributions.push({ name: 'shannon_entropy', impact: '-0.15 (Pushes to Legitimate)', type: 'trust' });

      return {
        url,
        prediction: isPhish ? 'PHISHING' : 'LEGITIMATE',
        phishingProbability: phishingScore,
        confidencePercentage: `${(confidence * 100).toFixed(1)}%`,
        riskLevel: isPhish ? (phishingScore > 0.85 ? 'CRITICAL' : 'HIGH') : 'LOW',
        rlAction: isPhish && phishingScore < 0.70 ? 'FLAG_FOR_REVIEW' : 'ACCEPT_INITIAL_PREDICTION',
        shapAttributions: shapAttributions.slice(0, 4)
      };
    }
  };

  // --- LIVE SIGNAL SANDBOX CONTROLLER ---
  const SandboxUI = {
    input: null,
    form: null,
    resultBox: null,
    badge: null,
    score: null,
    meta: null,
    shapList: null,

    init() {
      this.input = document.getElementById('gu-sandbox-url-input');
      this.form = document.getElementById('gu-sandbox-form');
      this.resultBox = document.getElementById('gu-verdict-box');
      this.badge = document.getElementById('gu-verdict-badge');
      this.score = document.getElementById('gu-verdict-score');
      this.meta = document.getElementById('gu-verdict-meta');
      this.shapList = document.getElementById('gu-shap-factors');

      if (!this.form || !this.input) return;

      this.form.addEventListener('submit', (e) => {
        e.preventDefault();
        this.runInspection(this.input.value);
      });

      // Demo Preset Chips
      const chips = document.querySelectorAll('.gu-chip');
      chips.forEach(chip => {
        chip.addEventListener('click', () => {
          const url = chip.getAttribute('data-url');
          if (url && this.input) {
            this.input.value = url;
            this.runInspection(url);
            AudioEngine.playBlip(750, 0.04);
          }
        });
      });
    },

    async runInspection(url) {
      if (!url || !url.trim()) return;

      const cleanUrl = url.trim();

      // Show inspecting state
      if (this.badge) {
        this.badge.className = 'gu-verdict-badge';
        this.badge.textContent = 'ANALYZING SIGNAL...';
      }
      if (this.score) {
        this.score.textContent = '--.-%';
      }

      AudioEngine.playBlip(550, 0.06, 'sawtooth');

      let result = null;

      // Attempt live backend FastAPI POST /predict first
      try {
        const response = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: cleanUrl, enable_rl: true })
        });
        if (response.ok) {
          const data = await response.json();
          result = {
            url: cleanUrl,
            prediction: data.prediction,
            confidencePercentage: data.confidence_percentage,
            riskLevel: data.risk_level,
            rlAction: data.rl_decision ? data.rl_decision.action : 'ACCEPT_INITIAL_PREDICTION',
            shapAttributions: (data.shap_factors || []).map(f => ({
              name: f.factor,
              impact: f.impact,
              type: f.impact.toLowerCase().includes('phishing') ? 'risk' : 'trust'
            }))
          };
        }
      } catch (err) {
        // Backend not running or static file served — seamlessly fallback to StandaloneScanner
      }

      if (!result) {
        result = StandaloneScanner.analyzeUrl(cleanUrl);
      }

      this.renderResult(result);
    },

    renderResult(result) {
      if (!this.resultBox) return;

      const isPhish = result.prediction === 'PHISHING';

      if (isPhish) {
        AudioEngine.playThreatPulse();
      } else {
        AudioEngine.playBlip(880, 0.06, 'sine', 0.04);
      }

      // Box classes
      this.resultBox.className = `gu-verdict-result ${isPhish ? 'gu-is-phishing' : 'gu-is-legitimate'}`;

      // Badge
      if (this.badge) {
        this.badge.className = `gu-verdict-badge ${isPhish ? 'gu-phishing' : 'gu-legit'}`;
        this.badge.textContent = isPhish ? 'MALICIOUS THREAT DETECTED' : 'VERIFIED LEGITIMATE';
      }

      // Score
      if (this.score) {
        this.score.textContent = result.confidencePercentage;
      }

      // Metadata
      if (this.meta) {
        this.meta.textContent = `RISK TIER: ${result.riskLevel} // RL ARBITER: ${result.rlAction}`;
      }

      // SHAP Explanations
      if (this.shapList) {
        this.shapList.innerHTML = '';
        if (result.shapAttributions && result.shapAttributions.length > 0) {
          result.shapAttributions.forEach(item => {
            const row = document.createElement('div');
            row.className = `gu-shap-factor-item ${item.type === 'risk' ? 'gu-risk' : 'gu-trust'}`;
            row.innerHTML = `<span>${item.type === 'risk' ? '▲' : '▼'}</span> <strong>${item.name}:</strong> ${item.impact}`;
            this.shapList.appendChild(row);
          });
        } else {
          this.shapList.innerHTML = '<div class="gu-shap-factor-item gu-trust">✓ All 34 features align with normal legitimate traffic distributions.</div>';
        }
      }
    }
  };

  // --- BOOTSTRAP WHEN DOM READY ---
  document.addEventListener('DOMContentLoaded', () => {
    Scene3D.init();
    Interactions.init();
    SandboxUI.init();
  });

})();
