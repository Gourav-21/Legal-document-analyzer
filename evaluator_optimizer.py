"""
Evaluator-Optimizer Pattern for Legal Document Analysis
Based on: https://dylancastillo.co/til/evaluator-optimizer-pydantic-ai.html
"""

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.settings import ModelSettings
from typing import List, Dict, Optional, Literal
import os
from dotenv import load_dotenv

load_dotenv()

# Evaluation Models
class LegalAnalysisEvaluation(BaseModel):
    """Evaluation criteria for legal analysis quality"""
    factual_accuracy: float = Field(..., ge=0, le=1, description="How accurate are the facts extracted from documents (0-1)")
    legal_compliance: float = Field(..., ge=0, le=1, description="How well does the analysis comply with Israeli labor law (0-1)")
    calculation_accuracy: float = Field(..., ge=0, le=1, description="How accurate are the monetary calculations (0-1)")
    completeness: float = Field(..., ge=0, le=1, description="How complete is the analysis coverage (0-1)")
    citation_quality: float = Field(..., ge=0, le=1, description="How well are laws and precedents cited (0-1)")
    clarity: float = Field(..., ge=0, le=1, description="How clear and understandable is the analysis (0-1)")
    
    overall_score: float = Field(..., ge=0, le=1, description="Overall quality score (0-1)")
    
    improvement_suggestions: List[str] = Field(
        default_factory=list,
        description="Specific suggestions for improving the analysis"
    )
    
    critical_issues: List[str] = Field(
        default_factory=list,
        description="Critical issues that must be addressed"
    )

class OptimizedAnalysis(BaseModel):
    """Optimized legal analysis output"""
    improved_analysis: str = Field(..., description="The improved legal analysis text")
    changes_made: List[str] = Field(default_factory=list, description="List of improvements made")
    confidence_level: Literal["high", "medium", "low"] = Field(..., description="Confidence in the analysis")

class EvaluatorOptimizer:
    """Evaluator-Optimizer system for legal document analysis"""
    
    def __init__(self):
        # Initialize Google Gemini model
        gemini_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_CLOUD_VISION_API_KEY")
        if not gemini_api_key:
            raise Exception("GOOGLE_API_KEY or GOOGLE_CLOUD_VISION_API_KEY must be set")
        
        google_provider = GoogleProvider(api_key=gemini_api_key)
        self.model = GoogleModel('gemini-2.5-pro', provider=google_provider)
        
        # Initialize evaluator agent
        self.evaluator = Agent(
            model=self.model,
            output_type=LegalAnalysisEvaluation,
            system_prompt="""You are an expert legal analysis evaluator specializing in Israeli labor law.

Your task is to evaluate the quality of legal document analysis based on these criteria:

🔍 **EVALUATION GUIDELINES (Same as review_analysis):**

1. בדוק בקפידה האם הניתוח **נכון, שלם, ומבוסס אך ורק** על:
   - החוקים שסופקו
   - פסקי הדין שסופקו
   - המסמכים שסופקו (תלוש שכר, חוזה עבודה, דוח נוכחות)

2. אין להשתמש בידע כללי או חיצוני, ואין להניח מידע שאינו מופיע במפורש במסמכים.

3. אין להעתיק או לעשות שימוש בערכים מהחוקים כדוגמה (כגון 6000 ₪, 186 שעות, 5 ימי מחלה), אלא אם הם מופיעים במפורש במסמכים.

4. אם לא מצוינים נתונים מפורשים (כמו שעות נוספות, ימי מחלה, סכום שכר מינימום), יש להניח **שאין עבירה**, ואין לדווח על הפרה או לבצע חישובים משוערים.

5. כל החישובים (סכומים, אחוזים, טבלאות) חייבים להיות מדויקים ולבוסס אך ורק על הערכים המופיעים במסמכים שסופקו.

6. יש לוודא שהמסקנות המשפטיות תואמות את המסמכים ואת החוק, ללא שגיאות, וללא חריגות או תוספות לא מבוססות.

🚫 **VERY IMPORTANT:**
- Do not recalculate data like wages per hour, overtime hours, sick days, etc. unless the document provides exact values.
- Do not infer or estimate violations without clear proof in the payslip.
- Use **only** the documents provided (e.g., payslip data, employment contracts data, and attendance records data). **Do not extract or reuse any example values (e.g., 6000 ₪, 186 hours, 14 hours overtime) that appear in the legal texts or examples.**
- Do **not invent** missing data. If the document does not include sufficient detail for a violation (e.g., no overtime hours), **do not report a violation**.
- Do not hallucinate sick days, overtime hours, or absences
- Think step by step and analyze the documents carefully. Do not rush to conclusions.
- While calculating read the whole law and dont miss anything and explain the calculations step by step and how you arrived at the final amounts.

**SCORING CRITERIA:**

1. **Factual Accuracy (0-1)**: How accurately are facts extracted from the source documents?
   - 1.0: All facts perfectly match source documents, no invented data
   - 0.8: Minor inaccuracies that don't affect conclusions
   - 0.6: Some factual errors that could impact analysis
   - 0.4: Multiple factual errors or some invented data
   - 0.2: Significant factual inaccuracies or hallucinated information
   - 0.0: Completely inaccurate facts or mostly invented data

2. **Legal Compliance (0-1)**: How well does the analysis comply with Israeli labor law?
   - 1.0: Perfect compliance, only reports violations with clear proof
   - 0.8: Minor compliance issues, mostly accurate legal interpretation
   - 0.6: Some compliance concerns or unsupported violation claims
   - 0.4: Multiple compliance issues or several unfounded violations
   - 0.2: Significant compliance problems or many false violations
   - 0.0: Major legal compliance failures or completely wrong violations

3. **Calculation Accuracy (0-1)**: How accurate are monetary calculations?
   - 1.0: All calculations mathematically correct and based on document data
   - 0.8: Minor calculation errors that don't affect main conclusions
   - 0.6: Some calculation mistakes or use of estimated values
   - 0.4: Multiple calculation errors or significant estimation
   - 0.2: Significant calculation problems or mostly estimated data
   - 0.0: Major calculation failures or completely invented numbers

4. **Completeness (0-1)**: How complete is the analysis coverage?
   - 1.0: Covers all relevant aspects present in documents
   - 0.8: Minor gaps in coverage of document-supported issues
   - 0.6: Some important document-supported aspects missing
   - 0.4: Multiple gaps in analysis of available data
   - 0.2: Significant omissions of clear document evidence
   - 0.0: Major aspects completely missing despite clear evidence

5. **Citation Quality (0-1)**: How well are laws and precedents cited?
   - 1.0: Perfect citations with accurate references to provided laws
   - 0.8: Minor citation issues but mostly accurate
   - 0.6: Some citation problems or missing references
   - 0.4: Multiple citation errors or poor referencing
   - 0.2: Poor citation quality or mostly unsupported claims
   - 0.0: No proper citations or completely unsupported analysis

6. **Clarity (0-1)**: How clear and understandable is the analysis?
   - 1.0: Perfectly clear, well-structured, step-by-step reasoning
   - 0.8: Minor clarity issues but generally well-organized
   - 0.6: Some unclear sections or confusing explanations
   - 0.4: Multiple clarity problems or poor organization
   - 0.2: Difficult to understand or poorly structured
   - 0.0: Very unclear, confusing, or incomprehensible

Calculate the overall_score as a weighted average:
- factual_accuracy: 25%
- legal_compliance: 25%
- calculation_accuracy: 20%
- completeness: 15%
- citation_quality: 10%
- clarity: 5%

Provide specific, actionable improvement suggestions and identify any critical issues that must be addressed.
Always respond in Hebrew when providing suggestions and critical issues.
""",
        )
        
        # Initialize optimizer agent
        self.optimizer = Agent(
            model=self.model,
            output_type=OptimizedAnalysis,
            system_prompt="""You are an expert legal analysis optimizer specializing in Israeli labor law.

Your task is to improve legal document analysis based on evaluation feedback using the same strict guidelines as the review_analysis method.

🔍 **OPTIMIZATION GUIDELINES (Same as review_analysis):**

1. בדוק בקפידה האם הניתוח **נכון, שלם, ומבוסס אך ורק** על:
   - החוקים שסופקו
   - פסקי הדין שסופקו
   - המסמכים שסופקו (תלוש שכר, חוזה עבודה, דוח נוכחות)

2. אין להשתמש בידע כללי או חיצוני, ואין להניח מידע שאינו מופיע במפורש במסמכים.

3. אין להעתיק או לעשות שימוש בערכים מהחוקים כדוגמה (כגון 6000 ₪, 186 שעות, 5 ימי מחלה), אלא אם הם מופיעים במפורש במסמכים.

4. אם לא מצוינים נתונים מפורשים (כמו שעות נוספות, ימי מחלה, סכום שכר מינימום), יש להניח **שאין עבירה**, ואין לדווח על הפרה או לבצע חישובים משוערים.

5. כל החישובים (סכומים, אחוזים, טבלאות) חייבים להיות מדויקים ולבוסס אך ורק על הערכים המופיעים במסמכים שסופקו.

6. יש לוודא שהמסקנות המשפטיות תואמות את המסמכים ואת החוק, ללא שגיאות, וללא חריגות או תוספות לא מבוססות.

🚫 **VERY IMPORTANT:**
- Do not recalculate data like wages per hour, overtime hours, sick days, etc. unless the document provides exact values.
- Do not infer or estimate violations without clear proof in the payslip.
- Use **only** the documents provided (e.g., payslip data, employment contracts data, and attendance records data). **Do not extract or reuse any example values (e.g., 6000 ₪, 186 hours, 14 hours overtime) that appear in the legal texts or examples.**
- Do **not invent** missing data. If the document does not include sufficient detail for a violation (e.g., no overtime hours), **do not report a violation**.
- Do not hallucinate sick days, overtime hours, or absences
- Think step by step and analyze the documents carefully. Do not rush to conclusions.
- While calculating read the whole law and dont miss anything and explain the calculations step by step and how you arrived at the final amounts.

**OPTIMIZATION PRIORITIES:**

1. **Address Critical Issues First**: Fix any critical legal or factual errors
2. **Remove Invented Data**: Eliminate any facts not present in source documents
3. **Fix Unfounded Violations**: Remove violation claims without clear document evidence
4. **Correct Calculations**: Fix mathematical errors using only document-provided values
5. **Enhance Legal Compliance**: Ensure strict adherence to provided laws and precedents
6. **Improve Citations**: Ensure proper citation of relevant laws and precedents
7. **Enhance Clarity**: Make the analysis clearer while maintaining accuracy

**CRITICAL REQUIREMENTS:**
- Use ONLY information from the provided source documents
- Do NOT invent or assume data not present in documents
- Base all legal conclusions on the provided laws and precedents
- Ensure all calculations are mathematically accurate and document-based
- Remove any violations that cannot be proven from the documents
- Maintain the original analysis structure and format
- Write in Hebrew as required

🛠 **OUTPUT REQUIREMENTS:**
- If the analysis is correct, return it as-is with minimal changes
- If there are errors, generate a corrected analysis that is fully compliant and mathematically accurate
- Never mention that it was revised, never explain what was fixed
- Simply return the corrected analysis as if it was always correct
- Always respond in Hebrew

Always explain what changes you made and why in the changes_made field.""",
        )
    
    async def evaluate_analysis(
        self,
        analysis: str,
        source_documents: Dict[str, str],
        laws: str,
        judgements: str,
        context: str = ""
    ) -> LegalAnalysisEvaluation:
        """Evaluate the quality of a legal analysis"""
        
        evaluation_prompt = f"""
הנך מקבל את החוקים, פסקי הדין, והניתוח המשפטי הבא לבדיקה ולהערכה:

📄 **חוקים:**
{laws}

📚 **פסקי דין:**
{judgements}

📑 **ניתוח משפטי להערכה:**
{analysis}

🧾 **מסמכים שסופקו לבדיקה:**
{chr(10).join([f"{doc_type.upper()} CONTENT:{chr(10)}{content}" for doc_type, content in source_documents.items() if content and content.strip() != 'Not provided'])}

🔍 **הקשר נוסף:**
{context if context else 'לא סופק הקשר נוסף'}

🔍 **הנחיות להערכה:**

בדוק בקפידה את הניתוח המשפטי על פי הקריטריונים הבאים:

1. **דיוק עובדתי**: האם כל העובדות מבוססות אך ורק על המסמכים שסופקו?
2. **תאימות משפטית**: האם הניתוח תואם את החוקים ופסקי הדין שסופקו?
3. **דיוק חישובים**: האם כל החישובים מדויקים ומבוססים על נתוני המסמכים?
4. **שלמות**: האם הניתוח מכסה את כל הנושאים הרלוונטיים הקיימים במסמכים?
5. **איכות ציטוטים**: האם החוקים ופסקי הדין מצוטטים נכון?
6. **בהירות**: האם הניתוח ברור ומובן?

⚠️ **שים לב במיוחד ל:**
- האם יש נתונים שהומצאו או הוערכו שלא על בסיס המסמכים?
- האם יש דיווח על הפרות ללא הוכחה ברורה במסמכים?
- האם החישובים מדויקים ומבוססים על הנתונים בפועל?
- האם נעשה שימוש בערכי דוגמה מהחוקים במקום נתוני המסמכים?

הערך את הניתוח ותן ציונים מדויקים עם הצעות שיפור ספציפיות.
"""
        
        result = await self.evaluator.run(evaluation_prompt, model_settings=ModelSettings(temperature=0.1))
        return result.output if hasattr(result, 'output') else result
    
    async def optimize_analysis(
        self,
        original_analysis: str,
        evaluation: LegalAnalysisEvaluation,
        source_documents: Dict[str, str],
        laws: str,
        judgements: str,
        context: str = ""
    ) -> OptimizedAnalysis:
        """Optimize the analysis based on evaluation feedback"""
        
        optimization_prompt = f"""
הנך מקבל את החוקים, פסקי הדין, והניתוח המשפטי הבא לשיפור על בסיס המשוב שהתקבל:

📄 **חוקים:**
{laws}

📚 **פסקי דין:**
{judgements}

📑 **ניתוח משפטי מקורי:**
{original_analysis}

📊 **משוב הערכה:**
ציון כללי: {evaluation.overall_score:.2f}
דיוק עובדתי: {evaluation.factual_accuracy:.2f}
תאימות משפטית: {evaluation.legal_compliance:.2f}
דיוק חישובים: {evaluation.calculation_accuracy:.2f}
שלמות: {evaluation.completeness:.2f}
איכות ציטוטים: {evaluation.citation_quality:.2f}
בהירות: {evaluation.clarity:.2f}

💡 **הצעות שיפור:**
{chr(10).join(f"- {suggestion}" for suggestion in evaluation.improvement_suggestions)}

⚠️ **בעיות קריטיות לטיפול:**
{chr(10).join(f"- {issue}" for issue in evaluation.critical_issues)}

🧾 **מסמכים לעיון:**
{chr(10).join([f"{doc_type.upper()} CONTENT:{chr(10)}{content}" for doc_type, content in source_documents.items() if content and content.strip() != 'Not provided'])}

🔍 **הקשר נוסף:**
{context if context else 'לא סופק הקשר נוסף'}

🔍 **הנחיות לשיפור:**

1. בדוק בקפידה האם הניתוח **נכון, שלם, ומבוסס אך ורק** על:
   - החוקים שסופקו
   - פסקי הדין שסופקו
   - המסמכים שסופקו (תלוש שכר, חוזה עבודה, דוח נוכחות)

2. אין להשתמש בידע כללי או חיצוני, ואין להניח מידע שאינו מופיע במפורש במסמכים.

3. אין להעתיק או לעשות שימוש בערכים מהחוקים כדוגמה (כגון 6000 ₪, 186 שעות, 5 ימי מחלה), אלא אם הם מופיעים במפורש במסמכים.

4. אם לא מצוינים נתונים מפורשים (כמו שעות נוספות, ימי מחלה, סכום שכר מינימום), יש להניח **שאין עבירה**, ואין לדווח על הפרה או לבצע חישובים משוערים.

5. כל החישובים (סכומים, אחוזים, טבלאות) חייבים להיות מדויקים ולבוסס אך ורק על הערכים המופיעים במסמכים שסופקו.

6. יש לוודא שהמסקנות המשפטיות תואמות את המסמכים ואת החוק, ללא שגיאות, וללא חריגות או תוספות לא מבוססות.

🛠 **אם הניתוח תקין** – החזר אותו כפי שהוא עם שינויים מינימליים.
🛠 **אם יש בו שגיאות** – תקן אותו כך שיהיה מדויק, מבוסס אך ורק על המסמכים, החוקים ופסקי הדין שסופקו.

**אין:**
- לציין שבוצע תיקון
- להסביר מה שונה
- להזכיר שהניתוח תוקן או נערך מחדש

✅ **יש להחזיר תמיד את הניתוח הסופי בלבד, בעברית מלאה וללא כל הערה.**

טפל בכל נקודות המשוב תוך שמירה על דיוק ותאימות משפטית.
"""
        
        result = await self.optimizer.run(optimization_prompt, model_settings=ModelSettings(temperature=0.2))
        return result.output if hasattr(result, 'output') else result
    
    async def evaluate_and_optimize(
        self,
        analysis: str,
        source_documents: Dict[str, str],
        laws: str,
        judgements: str,
        context: str = "",
        max_iterations: int = 3,
        target_score: float = 0.85
    ) -> tuple[str, LegalAnalysisEvaluation, List[str]]:
        """
        Iteratively evaluate and optimize analysis until target quality is reached
        
        Returns:
            - Final optimized analysis
            - Final evaluation
            - List of all changes made
        """
        
        current_analysis = analysis
        all_changes = []
        
        for iteration in range(max_iterations):
            print(f"🔄 Evaluation iteration {iteration + 1}/{max_iterations}")
            
            # Evaluate current analysis
            evaluation = await self.evaluate_analysis(
                current_analysis, source_documents, laws, judgements, context
            )
            
            print(f"📊 Current score: {evaluation.overall_score:.2f}")
            
            # Check if we've reached the target quality
            if evaluation.overall_score >= target_score:
                print(f"✅ Target score {target_score} reached!")
                break
            
            # Check if this is the last iteration
            if iteration == max_iterations - 1:
                print(f"⚠️ Max iterations reached. Final score: {evaluation.overall_score:.2f}")
                break
            
            # Optimize the analysis
            print("🔧 Optimizing analysis...")
            optimization_result = await self.optimize_analysis(
                current_analysis, evaluation, source_documents, laws, judgements, context
            )
            
            current_analysis = optimization_result.improved_analysis
            all_changes.extend(optimization_result.changes_made)
            
            print(f"✨ Changes made: {len(optimization_result.changes_made)}")
        
        return current_analysis, evaluation, all_changes

# Convenience function for easy integration
async def optimize_legal_analysis(
    analysis: str,
    source_documents: Dict[str, str],
    laws: str,
    judgements: str,
    context: str = "",
    target_score: float = 0.85
) -> Dict[str, any]:
    """
    Convenience function to optimize legal analysis using enhanced evaluation criteria
    
    Uses the same strict guidelines as review_analysis method:
    - Only uses data from provided documents
    - Does not invent or estimate missing data
    - Only reports violations with clear proof
    - Ensures mathematical accuracy based on document values
    - Maintains strict legal compliance
    
    Returns a dictionary with optimized analysis and metadata
    """
    
    optimizer = EvaluatorOptimizer()
    
    optimized_analysis, final_evaluation, changes = await optimizer.evaluate_and_optimize(
        analysis=analysis,
        source_documents=source_documents,
        laws=laws,
        judgements=judgements,
        context=context,
        target_score=target_score
    )
    
    return {
        'optimized_analysis': optimized_analysis,
        'quality_score': final_evaluation.overall_score,
        'detailed_scores': {
            'factual_accuracy': final_evaluation.factual_accuracy,
            'legal_compliance': final_evaluation.legal_compliance,
            'calculation_accuracy': final_evaluation.calculation_accuracy,
            'completeness': final_evaluation.completeness,
            'citation_quality': final_evaluation.citation_quality,
            'clarity': final_evaluation.clarity
        },
        'improvements_made': changes,
        'remaining_suggestions': final_evaluation.improvement_suggestions,
        'critical_issues': final_evaluation.critical_issues
    }