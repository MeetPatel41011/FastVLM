"use client"

import { useEffect, useState, useRef } from 'react'
import styles from './page.module.css'
import FastVLMCore from './FastVLMCore'

export default function Home() {
  const [index, setIndex] = useState(0)
  const [videoScale, setVideoScale] = useState(0.5)
  const videoBoxRef = useRef<HTMLDivElement>(null)
  
  const prefix = "Built for developers in the "
  const highlight = "agent-first"
  const suffix = " era"
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
          <p className={styles.subtitle}>Build the new way</p>
          <div className={styles.ctaContainer}>
            <button className={styles.primaryButton}>Download Antigravity</button>
            <button className={styles.secondaryButton}>See Overview</button>
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
    </main>
  )
}
