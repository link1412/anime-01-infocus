#version 410
in vec2 vP; flat in float vS,vCoc,vHalf,vCell,vCell2; flat in vec3 vC; out vec4 o;
uniform sampler2D uAtlas; uniform float uPxPerUnit,uGrid,uQuant,uFlying;
float glyph(float cell,vec2 q,float lod){
  vec2 c=vec2(mod(cell,uGrid),floor(cell/uGrid));
  vec2 st=(c+q*.5+.5)/uGrid;
  st.y=1.-st.y+(0.);
  return textureLod(uAtlas,st,lod).r;
}
void main(){
  float gpx=vS*uPxPerUnit;           // glyph half size in px
  float cpx=vCoc*uPxPerUnit;         // blur radius in px
  float w=uFlying>.5?smoothstep(.22,.62,vCoc/max(vS,1e-4)):smoothstep(.55,1.5,vCoc/max(vS,1e-4));
  float area=4.*vS*vS*uPxPerUnit*uPxPerUnit*.30;
  float val=0.;
  if(w<1.){
    vec2 q=vP/vS*vec2(1.,-1.);
    if(abs(q.x)<1.&&abs(q.y)<1.){
      float texels=64.*2./max(gpx*2.,1.);        // atlas texels per screen px
      float lod=log2(max(texels*max(cpx*.8,.6),.25));
      if(uQuant>0.){ q=(floor(q*uQuant)+.5)/uQuant; lod=0.; }   // magnified far enough, a character is screen pixels
      if(uFlying>.5) lod=min(lod,2.6);
      float g=glyph(vCell,q,lod);
      // lift the small end: tiny glyphs must stay bright, not grey mush
      val=g/max(area,1.);
    }
  }
  float disc=0.;
  if(w>0.){
    float d=length(vP)/max(vCoc,1e-4);
    float a=max(1.1/max(cpx,1.),.24*smoothstep(14.,50.,cpx));   // a big disc has a soft edge, or the edges stack into contour lines
    float rim=1.+.42*smoothstep(.62,.985,d)*(1.-.85*smoothstep(14.,50.,cpx));
    disc=(1.-smoothstep(1.-a,1.+a,d))*rim/(3.14159*cpx*cpx+1.);
  }
  float e=mix(val,disc,w);
  vec3 col=vC*e;
  // a sharp character is a thin bright stroke, not a clipped blob
  float top=max(col.r,max(col.g,col.b));
  float cap=mix(1.35,1e4,w);
  if(top>cap) col*=cap/top;
  o=vec4(col,1.);
}
