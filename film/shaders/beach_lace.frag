#version 410
// One seamless field for foam on a sheet of water: a fine cellular network of bubbles, thin strands that
// come and go, thickening into clots where the bubbles have gathered. Thresholding it high leaves clots
// alone, lower a broken lace, lower still a raft with holes: the whole life of a patch of foam is one number.
// Beside it two soft fields for where the rafts and leads lie, and the speckle of the bubbles themselves.
in vec2 uv; out vec4 o;
#include "common.glsl"
float pn(vec2 p,float P){
  vec2 i=floor(p),f=fract(p); f=f*f*(3.-2.*f);
  float a=hash21(mod(i,P)+P*.37),b=hash21(mod(i+vec2(1.,0.),P)+P*.37);
  float c=hash21(mod(i+vec2(0.,1.),P)+P*.37),d=hash21(mod(i+vec2(1.,1.),P)+P*.37);
  return mix(mix(a,b,f.x),mix(c,d,f.x),f.y);
}
float pf(vec2 x,float P){ return (pn(x*P,P)+.5*pn(x*P*2.,P*2.)+.25*pn(x*P*4.,P*4.))/1.75; }
float wallDist(vec2 g,float P,float seed){
  vec2 c=floor(g);
  float f1=9.,f2=9.;
  for(int j=-1;j<=1;j++) for(int i=-1;i<=1;i++){
    vec2 id=c+vec2(float(i),float(j));
    vec2 r=id+hash22(mod(id,P)+seed)-g;
    float d=dot(r,r);
    if(d<f1){ f2=f1; f1=d; } else if(d<f2) f2=d;
  }
  return sqrt(f2)-sqrt(f1);
}
float strand(float e,float w){ float a=e/w; return exp(-a*a); }
// The lace you can read from a beach, for the view facing the sun, over 16 m a tile. Seen this low only meshes a metre
// or two across survive foreshortening, so they lead: thin bands along curved walls, thicker where walls meet and
// where the foam has gathered, their edges bitten by small round holes, thin threads of the next size across the
// bigger meshes. Cut high it is a broken net, lower a raft with holes; inside a mesh there is nothing, which is what
// makes it lace and not fog.
uniform float uReadable;
vec2 cells(vec2 g,float P,float seed){
  vec2 c=floor(g);
  float f1=9.,f2=9.;
  for(int j=-1;j<=1;j++) for(int i=-1;i<=1;i++){
    vec2 id=c+vec2(float(i),float(j));
    vec2 r=id+hash22(mod(id,P)+seed)-g;
    float d=dot(r,r);
    if(d<f1){ f2=f1; f1=d; } else if(d<f2) f2=d;
  }
  return vec2(sqrt(f1),sqrt(f2)-sqrt(f1));
}
float readable(vec2 x){
  vec2 w0=vec2(pf(x,5.),pf(x+.37,5.))-.5;
  vec2 w1=vec2(pf(x+.5,17.),pf(x+.81,17.))-.5;
  vec2 w2=vec2(pf(x+.2,53.),pf(x+.67,53.))-.5;
  float nL=pf(x+.11,5.),nM=pf(x+.23,17.),nS=pf(x+.47,53.),nX=pf(x+.71,160.);
  float e1=cells((x+w0*.07+w1*.02)*9.,9.,1.).y;
  float e2=cells((x+w0*.03+w1*.016+w2*.004)*23.,23.,2.).y;
  vec2 c3=cells((x+w1*.008+w2*.005)*61.,61.,3.);
  vec2 c4=cells((x+w2*.004)*170.,170.,4.);
  float en=e1+.05*(nS-.5)+.03*(nX-.5);
  float band=exp(-en*en/(.17*.17))*(.55+.6*smoothstep(.28,.62,nL*.6+nM*.4));
  // bitten by holes it has not yet closed around
  float bite=mix(.62,1.,smoothstep(.1,.55,c3.x))*mix(.82,1.,smoothstep(.1,.5,c4.x));
  float en2=e2+.03*(nS-.5);
  float thread=.62*exp(-en2*en2/(.05*.05))*smoothstep(.36,.66,nM*.7+nL*.3)*mix(.7,1.,smoothstep(.1,.5,c4.x));
  // from a band into its mesh the net gets finer and finer before it stops
  float skirt=exp(-en*en/(.34*.34));
  float fine=skirt*(.6*exp(-c3.y*c3.y/(.11*.11))*smoothstep(.3,.55,nS*.5+nM*.5)+.42*exp(-c4.y*c4.y/(.17*.17))*smoothstep(.34,.6,nX*.5+nS*.5));
  // the raft closing in: first the small meshes along the walls, then the big ones from their corners
  float raft=.3*(1.-smoothstep(.0,.62,e1))*mix(.55,1.,smoothstep(.05,.3,c3.y))*mix(.75,1.,smoothstep(.04,.22,c4.y))+.17*nM+.08*nS;
  // and nothing is solid: a raft is bubbles gathered round holes it has not closed yet
  return (max(band*bite,max(thread,fine))+raft)*mix(.55,1.,smoothstep(.12,.46,c3.x))*mix(.74,1.,smoothstep(.1,.5,c4.x));
}
void main(){
  vec2 x=uv;
  if(uReadable>.5){
    o=vec4(pf(x+.13,3.),pf(x+.29,11.),readable(x),pf(x+.53,307.)*.5+strand(wallDist(x*450.,450.,5.),.26)*.5);
    return;
  }
  vec2 w0=vec2(pf(x,6.),pf(x+.37,6.))-.5;
  vec2 w1=vec2(pf(x,23.),pf(x+.61,23.))-.5;
  vec2 w2=vec2(pf(x,71.),pf(x+.19,71.))-.5;
  float nL=pf(x+.11,9.),nM=pf(x+.23,27.),nS=pf(x+.47,81.),nX=pf(x+.71,243.);
  float e1=wallDist((x+w0*.03)*14.,14.,1.);
  float e2=wallDist((x+w0*.014+w1*.012)*41.,41.,2.);
  float e3=wallDist((x+w1*.005+w2*.006)*113.,113.,3.);
  float e4=wallDist((x+w2*.002)*331.,331.,4.);
  float gone2=smoothstep(.36,.62,nM*.55+nS*.45);
  float gone3=smoothstep(.3,.6,nS*.6+nX*.4);
  float phi=.14*strand(e1,.045)*smoothstep(.42,.7,nL)
           +.46*strand(e2,.07+.07*nM)*(.28+1.05*gone2)
           +.34*strand(e3,.1+.1*nS)*(.2+1.05*gone3)
           +.20*strand(e4,.2)*(.35+.9*nX)
           +.34*nL+.24*nM+.12*nS;
  o=vec4(pf(x+.13,3.),pf(x+.29,11.),phi,pf(x+.53,307.)*.5+strand(e4,.26)*.5);
}
