(function(){
'use strict';
var toggle=document.getElementById('menu-toggle');
var nav=document.getElementById('site-nav');
function setMenu(open){if(!nav||!toggle)return;nav.classList.toggle('open',open);toggle.setAttribute('aria-expanded',String(open));toggle.setAttribute('aria-label',open?'Close navigation menu':'Open navigation menu');toggle.textContent=open?'×':'☰';}
if(toggle)toggle.addEventListener('click',function(){setMenu(toggle.getAttribute('aria-expanded')!=='true');});
if(nav)nav.querySelectorAll('a').forEach(function(link){link.addEventListener('click',function(){setMenu(false);});});
document.addEventListener('keydown',function(ev){if(ev.key==='Escape')setMenu(false);});
var copyButtons=document.querySelectorAll('button.copy');
copyButtons.forEach(function(btn){btn.addEventListener('click',async function(){
var code=btn.closest('.pre-wrap').querySelector('pre code');if(!code)return;
var content=code.textContent;var copied=false;
try{if(navigator.clipboard&&window.isSecureContext){await navigator.clipboard.writeText(content);copied=true;}}catch(_e){}
if(!copied){var select=window.getSelection();var range=document.createRange();range.selectNodeContents(code);select.removeAllRanges();select.addRange(range);}
var prior=btn.textContent;btn.textContent=copied?'Copied':'Selected';btn.setAttribute('aria-label',copied?'Copied to clipboard':'Command selected for manual copying');
window.setTimeout(function(){btn.textContent=prior;btn.setAttribute('aria-label','Copy code');},1800);
});});
})();
