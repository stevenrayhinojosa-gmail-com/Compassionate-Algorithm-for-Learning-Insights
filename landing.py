import streamlit as st

def main():
    
    # Main header
    st.title("🎓 Student Behavior Analysis Platform")
    st.subheader("AI-Powered Behavior Prediction & Wellness Management")
    
    # Hero section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Welcome to the Future of Student Support
        
        Our advanced machine learning platform helps educators and support staff:
        
        ✅ **Predict behavior patterns** before they become challenges  
        ✅ **Track wellness trends** with comprehensive data analysis  
        ✅ **Set up smart alerts** for early intervention  
        ✅ **Manage medications** and environmental factors  
        ✅ **Make data-driven decisions** for student success  
        
        Transform your approach to student wellness with intelligent insights that matter.
        """)
        
        # Call to action
        st.markdown("---")
        if st.button("🚀 Get Started with Behavior Analysis", type="primary", use_container_width=True):
            st.success("Redirecting to the analysis dashboard...")
            st.markdown("**Next:** You'll be taken to the main analysis page where you can:")
            st.markdown("- View behavior predictions")
            st.markdown("- Upload your data")
            st.markdown("- Configure alerts and settings")
    
    with col2:
        st.markdown("### 📊 Platform Features")
        st.info("**Real-time Predictions**\nGet tomorrow's behavior forecast")
        st.info("**Smart Alerts**\nReceive notifications when intervention is needed")
        st.info("**Data Upload**\nEasily import CSV files or Google Sheets")
        st.info("**Comprehensive Tracking**\nMonitor medications, environment, and more")
    
    # Benefits section
    st.markdown("---")
    st.header("🌟 Why Choose Our Platform?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎯 Proactive Approach**
        
        Stop reacting to behavioral challenges. Our AI predicts patterns before they escalate, giving you time to provide support when it matters most.
        """)
    
    with col2:
        st.markdown("""
        **📈 Evidence-Based Insights**
        
        Make decisions backed by data. Track progress over time and identify what interventions work best for each student.
        """)
    
    with col3:
        st.markdown("""
        **⚡ Easy Integration**
        
        Upload your existing data in minutes. Works with CSV files, Google Sheets, and other common formats.
        """)
    
    # Getting started section
    st.markdown("---")
    st.header("🚀 Ready to Begin?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **For First-Time Users:**
        1. Click "Get Started" above
        2. Enter student name
        3. Upload your behavior data
        4. View instant predictions
        """)
    
    with col2:
        st.markdown("""
        **Need Help?**
        - 📋 Sample data is included for testing
        - 🔧 Configure alerts in the settings
        - 📞 Support available for setup assistance
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("*Empowering educators with intelligent behavior analysis since 2024*")

if __name__ == "__main__":
    main()