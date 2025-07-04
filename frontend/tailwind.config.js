/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        'exxon-red': '#E31837',
        'exxon-dark-blue': '#003366',
        'exxon-light-blue': '#0066CC',
        'exxon-gray': '#F5F5F5',
        'exxon-dark-gray': '#333333',
        'exxon-text': '#2C2C2C',
      },
      fontFamily: {
        'inter': ['Inter', 'sans-serif'],
      },
    },
  },
  plugins: [],
};