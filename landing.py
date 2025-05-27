import streamlit as st

# Set page config
st.set_page_config(
    page_title="Student Behavior Analysis Platform",
    page_icon="📊",
    layout="wide"
)

def main():
    
    # Main header
    st.title("CALI: Compassionate Algorithm for Learning Insights")
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
        
        # Data requirements section
        st.markdown("### 📋 What Data Does CALI Need?")
        st.markdown("""
        Your behavior tracking data should include these three key components:
        
        • **Daily timestamps** - Dates when behaviors were recorded
        • **Time-slot behaviors** - Red/Yellow/Green markers throughout each day  
        • **Student information** - Basic details to track individual progress
        
        *Don't worry about formatting - CALI automatically cleans and processes messy data!*
        """)
        
        # Upload section
        st.markdown("---")
        st.markdown("### 📁 Quick Upload")
        st.markdown("Upload your behavior data file and let CALI automatically detect the student information:")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file", 
            type=['csv'],
            help="Upload your Applied Behavior Analysis data file"
        )
        
        if uploaded_file:
            try:
                import pandas as pd
                import tempfile
                import os
                
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Try to detect student name from filename or file content
                filename = uploaded_file.name
                student_name = "Unknown Student"
                
                # Extract from filename first
                if filename:
                    # Remove file extension and clean up
                    base_name = os.path.splitext(filename)[0]
                    # Look for common patterns like "John_Doe_data" or "Smith-behavior"
                    import re
                    name_patterns = [
                        r'([A-Za-z]+[_\-\s][A-Za-z]+)',  # FirstName_LastName or First-Last
                        r'([A-Za-z]+)',  # Single name
                    ]
                    
                    for pattern in name_patterns:
                        match = re.search(pattern, base_name)
                        if match:
                            student_name = match.group(1).replace('_', ' ').replace('-', ' ').title()
                            break
                
                # Try to read file content for additional clues
                try:
                    df = pd.read_csv(tmp_path, nrows=10)  # Read first few rows
                    # Look for student name in headers or first few cells
                    for col in df.columns:
                        if 'name' in str(col).lower() or 'student' in str(col).lower():
                            first_value = str(df[col].iloc[0]) if len(df) > 0 else ""
                            if first_value and first_value != 'nan' and len(first_value) > 1:
                                student_name = first_value
                                break
                except:
                    pass
                
                # Clean up temp file
                os.unlink(tmp_path)
                
                st.success(f"✅ File uploaded successfully!")
                st.info(f"🎓 Detected student: **{student_name}**")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🚀 Analyze This Data", type="primary"):
                        st.session_state['uploaded_file'] = uploaded_file
                        st.session_state['detected_student'] = student_name
                        st.success("Data ready for analysis! Switch to the main dashboard (port 5000) to view predictions.")
                
                with col2:
                    corrected_name = st.text_input("Correct student name if needed:", value=student_name)
                    if corrected_name != student_name:
                        st.session_state['detected_student'] = corrected_name
                        st.info(f"Updated to: {corrected_name}")
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.info("Please ensure your file is a valid CSV with behavior data.")
        
        # Alternative call to action
        st.markdown("---")
        if st.button("📊 Go to Full Dashboard", use_container_width=True):
            st.info("💡 **Dashboard Access:** Open the analysis dashboard on port 5000 to access all features including data upload, predictions, and settings.")
    
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