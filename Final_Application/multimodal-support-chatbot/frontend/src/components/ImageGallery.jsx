import React, { useState, useCallback } from 'react';
import { X, ZoomIn, ChevronLeft, ChevronRight } from 'lucide-react';

/**
 * ImageGallery — displays retrieved images inline with a lightbox modal.
 * Images are shown as cards; clicking opens a full-screen overlay.
 */
export default function ImageGallery({ images }) {
  const [lightboxIdx, setLightboxIdx] = useState(null);

  const openLightbox = useCallback((idx) => setLightboxIdx(idx), []);
  const closeLightbox = useCallback(() => setLightboxIdx(null), []);

  const goPrev = useCallback(() => {
    setLightboxIdx((prev) => (prev > 0 ? prev - 1 : images.length - 1));
  }, [images.length]);

  const goNext = useCallback(() => {
    setLightboxIdx((prev) => (prev < images.length - 1 ? prev + 1 : 0));
  }, [images.length]);

  // Keyboard navigation
  React.useEffect(() => {
    if (lightboxIdx === null) return;
    const handler = (e) => {
      if (e.key === 'Escape') closeLightbox();
      if (e.key === 'ArrowLeft') goPrev();
      if (e.key === 'ArrowRight') goNext();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [lightboxIdx, closeLightbox, goPrev, goNext]);

  if (!images || images.length === 0) return null;

  return (
    <>
      {/* Inline image cards */}
      <div className="image-gallery">
        {images.map((img, idx) => (
          <button
            key={img.image_id || idx}
            className="image-card"
            onClick={() => openLightbox(idx)}
            title="Click to enlarge"
          >
            <div className="image-card-inner">
              <img
                src={img.url || img.thumbnail_url}
                alt={img.caption || 'Retrieved image'}
                className="image-card-img"
                loading="lazy"
              />
              <div className="image-card-overlay">
                <ZoomIn className="w-5 h-5" />
              </div>
            </div>
            <div className="image-card-footer">
              <p className="image-caption">{img.caption || 'Image'}</p>
              <div className="image-meta">
                {img.source && (
                  <span className="image-source">{img.source}</span>
                )}
                {img.relevance_score != null && (
                  <span className="image-score">
                    {(img.relevance_score * 100).toFixed(0)}% match
                  </span>
                )}
              </div>
            </div>
          </button>
        ))}
      </div>

      {/* Lightbox modal */}
      {lightboxIdx !== null && images[lightboxIdx] && (
        <div className="lightbox-overlay" onClick={closeLightbox}>
          <div className="lightbox-content" onClick={(e) => e.stopPropagation()}>
            {/* Close button */}
            <button className="lightbox-close" onClick={closeLightbox}>
              <X className="w-5 h-5" />
            </button>

            {/* Navigation arrows */}
            {images.length > 1 && (
              <>
                <button className="lightbox-nav lightbox-nav--prev" onClick={goPrev}>
                  <ChevronLeft className="w-6 h-6" />
                </button>
                <button className="lightbox-nav lightbox-nav--next" onClick={goNext}>
                  <ChevronRight className="w-6 h-6" />
                </button>
              </>
            )}

            {/* Image */}
            <img
              src={images[lightboxIdx].url}
              alt={images[lightboxIdx].caption || 'Image'}
              className="lightbox-img"
            />

            {/* Caption bar */}
            <div className="lightbox-caption">
              <p className="lightbox-caption-text">
                {images[lightboxIdx].caption}
              </p>
              {images[lightboxIdx].source && (
                <span className="lightbox-source">
                  {images[lightboxIdx].source}
                </span>
              )}
              {images.length > 1 && (
                <span className="lightbox-counter">
                  {lightboxIdx + 1} / {images.length}
                </span>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
