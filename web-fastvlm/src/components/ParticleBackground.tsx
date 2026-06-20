"use client";

import { useEffect, useRef } from 'react';

export default function ParticleBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;
    
    // Explicitly declare particles array at the top so it's accessible by handleScroll
    let particles: Particle[] = [];
    
    // Mouse state tracking
    const mouse = {
      x: -1000,
      y: -1000,
      isMoving: false
    };

    let moveTimeout: NodeJS.Timeout;
    let lastScrollY = window.scrollY;
    let isBoxMax = false;

    const handleBoxScale = (e: any) => {
      isBoxMax = e.detail.isMax;
      if (isBoxMax) {
        // Immediately return dots to their original positions
        mouse.isMoving = false;
        clearTimeout(moveTimeout);
      }
    };
    
    // @ts-ignore - CustomEvent
    window.addEventListener('boxScaleState', handleBoxScale);

    const activateMotion = () => {
      if (isBoxMax || mouse.x === -1000) return; // Ignore tracking when video box is full screen
      mouse.isMoving = true;
      clearTimeout(moveTimeout);
      moveTimeout = setTimeout(() => {
        mouse.isMoving = false;
      }, 700); // Keep nodes gathered for 0.7 seconds after motion stops
    };

    const handleMouseMove = (e: MouseEvent) => {
      mouse.x = e.clientX;
      mouse.y = e.clientY;
      activateMotion();
    };

    const handleScroll = () => {
      // Just activate motion. The update() function will use the last known
      // mouse.x and mouse.y to attract the nodes, exactly like handleMouseMove.
      activateMotion();
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('scroll', handleScroll);

    class Particle {
      x: number;
      y: number;
      initialX: number;
      initialY: number;
      vx: number;
      vy: number;
      radius: number;

      constructor(x: number, y: number) {
        this.x = x;
        this.y = y;
        // Store the very initial spawn position
        this.initialX = x;
        this.initialY = y;
        this.vx = 0;
        this.vy = 0;
        this.radius = 2.5; 
      }

      update() {
        let targetVx = 0;
        let targetVy = 0;

        if (mouse.isMoving) {
          const dx = mouse.x - this.x;
          const dy = mouse.y - this.y;
          const distToMouse = Math.sqrt(dx * dx + dy * dy);

          const minRadius = 100; // Fixed distance from pointer
          const nodeSpacing = 25; // Distance between nodes
          
          const nx = distToMouse > 0 ? dx / distToMouse : 0;
          const ny = distToMouse > 0 ? dy / distToMouse : 0;
          
          // Attract ALL nodes on screen to the pointer
          let force = (distToMouse - minRadius) * 0.05; 
          
          targetVx = nx * force;
          targetVy = ny * force;

          // Node-to-node repulsion
          for (let i = 0; i < particles.length; i++) {
            const other = particles[i];
            if (other === this) continue;
            
            const ddx = this.x - other.x;
            const ddy = this.y - other.y;
            const distBetween = Math.sqrt(ddx * ddx + ddy * ddy);
            
            if (distBetween < nodeSpacing && distBetween > 0) {
              const repelForce = (nodeSpacing - distBetween) * 0.3;
              targetVx += (ddx / distBetween) * repelForce;
              targetVy += (ddy / distBetween) * repelForce;
            }
          }
        } else {
          // MOUSE STOPPED: Homing sequence to exact initial position
          const dxInit = this.initialX - this.x;
          const dyInit = this.initialY - this.y;
          const distToInit = Math.sqrt(dxInit * dxInit + dyInit * dyInit);

          // Stronger pull the further away it is, snapping into place when close
          if (distToInit > 0.5) {
            targetVx = dxInit * 0.08;
            targetVy = dyInit * 0.08;
          } else {
            // Lock in place when very close
            this.x = this.initialX;
            this.y = this.initialY;
            targetVx = 0;
            targetVy = 0;
          }
        }

        // Apply velocity with easing
        this.vx += (targetVx - this.vx) * 0.2;
        this.vy += (targetVy - this.vy) * 0.2;

        // Friction to slow down faster when arriving
        this.vx *= 0.9;
        this.vy *= 0.9;

        this.x += this.vx;
        this.y += this.vy;
      }

      draw() {
        if (!ctx) return;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
        ctx.fillStyle = '#3279F9'; // Google Blue
        ctx.fill();
      }
    }

    const initParticles = () => {
      particles = [];
      const density = 6000; 
      const numParticles = Math.floor((canvas.width * canvas.height) / density);
      
      for (let i = 0; i < numParticles; i++) {
        const x = Math.random() * canvas.width;
        const y = Math.random() * canvas.height;
        particles.push(new Particle(x, y));
      }
    };

    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      initParticles();
    };

    const animate = () => {
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      particles.forEach(p => {
        p.update();
        p.draw();
      });

      animationFrameId = requestAnimationFrame(animate);
    };

    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
    animate();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('scroll', handleScroll);
      // @ts-ignore
      window.removeEventListener('boxScaleState', handleBoxScale);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        zIndex: 0,
        pointerEvents: 'none',
        backgroundColor: '#FFFFFF',
      }}
    />
  );
}
