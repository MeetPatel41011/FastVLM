"use client"

import { useEffect, useState, useRef } from 'react'
import styles from './page.module.css'
import FastVLMCore from './FastVLMCore'

export default function Home() {
  const [index, setIndex] = useState(0)
  const [videoScale, setVideoScale] = useState(0.5)
  const videoBoxRef = useRef<HTMLDivElement>(null)
  
  const prefix = "FastVLM: Live Camera Q&A at "
  const highlight = "Zero Latency"
  const suffix = " on the Edge"
  const fullText = prefix + highlight + suffix
  
  useEffect(() => {
    // Typing Animation
    const interval = setInterval(() => {
      setIndex((prev) => {
        if (prev >= fullText.length) {
          clearInterval(interval)
          return prev
        }
        return prev + 1
      })
    }, 70) 

    // Scroll Animation for Video Box
    const handleScroll = () => {
      if (videoBoxRef.current) {
        const rect = videoBoxRef.current.getBoundingClientRect()
        const windowHeight = window.innerHeight
        
        // Progress: 0 when top of box is at the very bottom of the screen
        // 1 when the middle of the box reaches the middle of the screen
        const boxCenter = rect.top + (rect.height / 2);
        let progress = (windowHeight - rect.top) / (windowHeight / 1.2)
        progress = Math.max(0, Math.min(1, progress))
        
        // Map progress (0 -> 1) to scale (0.5 -> 1.0)
        // This makes it start at half size and grow to full size
        const newScale = 0.5 + (0.5 * progress)
        setVideoScale(newScale)
        
        // Notify ParticleBackground if the box is fully expanded
        window.dispatchEvent(new CustomEvent('boxScaleState', { detail: { isMax: newScale >= 1.0 } }))
      }
    }

    window.addEventListener('scroll', handleScroll)
    handleScroll() // Initialize on mount

    return () => {
      clearInterval(interval)
      window.removeEventListener('scroll', handleScroll)
    }
  }, [fullText.length])

  return (
    <main className={styles.main}>
      <section className={styles.hero}>
        <div className={styles.heroContent}>
          <h1 className={styles.title}>
            {fullText.slice(0, Math.min(index, prefix.length))}
            {index > prefix.length && (
              <span className={styles.highlight}>
                {fullText.slice(prefix.length, Math.min(index, prefix.length + highlight.length))}
              </span>
            )}
            {index > prefix.length + highlight.length && (
              <span>
                {fullText.slice(prefix.length + highlight.length, index)}
              </span>
            )}
            <span className={styles.cursor}>|</span>
          </h1>
          <p className={styles.subtitle}>A demonstration of end-to-end problem solving and modern AI architecture.</p>
          <div className={styles.ctaContainer}>
            <a href="https://github.com/MeetPatel41011/FastVLM" target="_blank" rel="noopener noreferrer">
              <button className={styles.primaryButton}>View Source Code</button>
            </a>
            <a href="https://www.linkedin.com/in/meetpatel41011/" target="_blank" rel="noopener noreferrer">
              <button className={styles.secondaryButton}>Connect on LinkedIn</button>
            </a>
          </div>
        </div>
      </section>

      {/* Animated Core Box (FastVLM System) */}
      <section className={styles.videoSection}>
        <div 
          ref={videoBoxRef}
          className={styles.videoBox}
          style={{ transform: `scale(${videoScale})`, overflow: 'hidden', display: 'flex' }}
        >
          <FastVLMCore />
        </div>
      </section>

      {/* Recruiter / Resume Section */}
      <section className={styles.aboutSection}>
        <div className={styles.aboutContent}>
          <h2>Why I Built This</h2>
          <p>
            I built FastVLM to demonstrate my core engineering philosophy: 
            <strong> taking a complex problem, chopping it down into manageable parts, and getting things done no matter what.</strong>
          </p>
          <p>
            This project required navigating cutting-edge local LLMs (Qwen2-VL), integrating live WebRTC camera feeds, 
            orchestrating a Python backend with a Next.js frontend, and deploying seamlessly to cloud GPUs on Modal. 
            When challenges arose—like hardware constraints or cross-origin latency—I broke them down, 
            researched the solutions, and engineered a zero-latency hybrid architecture that works natively in the browser.
          </p>
        </div>
      </section>
    </main>
  )
}
