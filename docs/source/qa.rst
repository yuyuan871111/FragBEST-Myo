Common questions when using FragBEST-Myo
========================================

1. What if I have skeletal myosin instead of cardiac? Should I make any modifications to the code in the tutorial?
------------------------------------------------------------------------------------------------------------------

**Answer:**  
You can follow `Tutorial 3 <notebooks/03_holo_like_form_detect.html>`_ as outlined. 
However, the key modification you need to make is ensuring that the **pocket center** 
is correctly defined. Since the reference structure (cardiac myosin) 
might not be suitable for skeletal myosin, you will need to use the 
**first frame** (``frame 0``) of your trajectory to define the pocket center 
and align the trajectory accordingly.

Here's what you need to do (also mentioned this within ``hint`` in 
`Tutorial 3 <notebooks/03_holo_like_form_detect.html>`_):   

1. Use the first frame of your trajectory to identify the pocket center.   
2. Align the entire trajectory to the first frame to ensure consistency.   

This adjustment ensures that the analysis is tailored to skeletal myosin 
or any other isoform you are working with.


2. I have other chains in the trajectory (not just the myosin heavy chain), should I remove them? 
-------------------------------------------------------------------------------------------------

**Answer:**   
It depends.  

If those chains are **relevant to or close to your region of interest 
(pocket)**, you should **keep them** to preserve the original local environment 
and ensure accurate analysis.  

However, if those chains are **far away from your pocket region** and do not 
interact with it, it is recommended to **remove them** to reduce computational 
cost and simplify the analysis. This can help streamline the workflow without 
compromising the accuracy of results.
